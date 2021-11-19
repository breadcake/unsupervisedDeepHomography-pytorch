import numpy as np
from utils.torch_spatial_transformer import transformer
import torch
from torch import nn
import torch.nn.functional as F

def L1_smooth_loss(x, y):
    abs_diff = torch.abs(x - y)
    abs_diff_lt_1 = torch.le(abs_diff, 1)
    return torch.mean(torch.where(abs_diff_lt_1, 0.5 * abs_diff ** 2, abs_diff - 0.5))

def SSIM_loss(x, y, size=3):
    # C = (K*L)^2 with K = max of intensity range (i.e. 255). L is very small
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, size, 1, padding=0)
    mu_y = F.avg_pool2d(y, size, 1, padding=0)

    sigma_x = F.avg_pool2d(x ** 2, size, 1, padding=0) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, size, 1, padding=0) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, size, 1, padding=0) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)

def NCC_loss(x, y):
    """Consider x, y are vectors. Take L2 of the difference
       of the them after being normalized by their length"""
    len_x = torch.sqrt(torch.sum(x ** 2))
    len_y = torch.sqrt(torch.sum(y ** 2))
    return torch.sqrt(torch.sum((x / len_x - y / len_y) ** 2))

class ConvBlock(nn.Module):
    def __init__(self, inchannels, outchannels, batch_norm=False, pool=True):
        super(ConvBlock, self).__init__()
        layers = []
        layers.append(nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        if batch_norm:
            layers.append(nn.BatchNorm2d(outchannels))
        layers.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        if batch_norm:
            layers.append(nn.BatchNorm2d(outchannels))
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class HomographyModel(nn.Module):
    def __init__(self, batch_norm=False):
        super(HomographyModel, self).__init__()
        self.feature = nn.Sequential(
            ConvBlock(2, 64, batch_norm),
            ConvBlock(64, 64, batch_norm),
            ConvBlock(64, 128, batch_norm),
            ConvBlock(128, 128, batch_norm, pool=False),
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 16 * 16, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 8)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, I1_aug, I2_aug, I_aug, h4p, gt, patch_indices):
        batch_size, _, img_h, img_w = I_aug.size()
        _, _, patch_size, patch_size = I1_aug.size()

        y_t = torch.arange(0, batch_size * img_w * img_h,
                           img_w * img_h)
        batch_indices_tensor = y_t.unsqueeze(1).expand(y_t.shape[0], patch_size * patch_size).reshape(-1)

        M_tensor = torch.tensor([[img_w / 2.0, 0., img_w / 2.0],
                                 [0., img_h / 2.0, img_h / 2.0],
                                 [0., 0., 1.]])

        if torch.cuda.is_available():
            M_tensor = M_tensor.cuda()
            batch_indices_tensor = batch_indices_tensor.cuda()

        M_tile = M_tensor.unsqueeze(0).expand(batch_size, M_tensor.shape[-2], M_tensor.shape[-1])

        # Inverse of M
        M_tensor_inv = torch.inverse(M_tensor)
        M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, M_tensor_inv.shape[-2],
                                                      M_tensor_inv.shape[-1])

        pred_h4p = self.build_model(I1_aug, I2_aug)

        H_mat = self.solve_DLT(h4p, pred_h4p).squeeze(1)

        pred_I2 = self.transform(patch_size, M_tile_inv, H_mat, M_tile,
                                 I_aug, patch_indices, batch_indices_tensor)

        h_loss = torch.sqrt(torch.mean((pred_h4p - gt) ** 2))
        rec_loss, ssim_loss, l1_loss, l1_smooth_loss, ncc_loss = self.build_losses(pred_I2, I2_aug)

        out_dict = {}
        out_dict.update(h_loss=h_loss, rec_loss=rec_loss, ssim_loss=ssim_loss, l1_loss=l1_loss,
                        l1_smooth_loss=l1_smooth_loss, ncc_loss=ncc_loss,
                        pred_h4p=pred_h4p, H_mat=H_mat, pred_I2=pred_I2)

        return out_dict

    def build_model(self, I1_aug, I2_aug):
        model_input = torch.cat([I1_aug, I2_aug], dim=1)
        x = self.feature(model_input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def solve_DLT(self, src_p, off_set):
        # src_p: shape=(bs, n, 4, 2)
        # off_set: shape=(bs, n, 4, 2)
        # can be used to compute mesh points (multi-H)

        bs, _ = src_p.shape
        divide = int(np.sqrt(len(src_p[0]) / 2) - 1)
        row_num = (divide + 1) * 2

        for i in range(divide):
            for j in range(divide):

                h4p = src_p[:, [2 * j + row_num * i, 2 * j + row_num * i + 1,
                                2 * (j + 1) + row_num * i, 2 * (j + 1) + row_num * i + 1,
                                2 * (j + 1) + row_num * i + row_num, 2 * (j + 1) + row_num * i + row_num + 1,
                                2 * j + row_num * i + row_num, 2 * j + row_num * i + row_num + 1]].reshape(bs, 1, 4, 2)

                pred_h4p = off_set[:, [2 * j + row_num * i, 2 * j + row_num * i + 1,
                                       2 * (j + 1) + row_num * i, 2 * (j + 1) + row_num * i + 1,
                                       2 * (j + 1) + row_num * i + row_num, 2 * (j + 1) + row_num * i + row_num + 1,
                                       2 * j + row_num * i + row_num, 2 * j + row_num * i + row_num + 1]].reshape(bs, 1,
                                                                                                                  4, 2)

                if i + j == 0:
                    src_ps = h4p
                    off_sets = pred_h4p
                else:
                    src_ps = torch.cat((src_ps, h4p), axis=1)
                    off_sets = torch.cat((off_sets, pred_h4p), axis=1)

        bs, n, h, w = src_ps.shape

        N = bs * n

        src_ps = src_ps.reshape(N, h, w)
        off_sets = off_sets.reshape(N, h, w)

        dst_p = src_ps + off_sets

        ones = torch.ones(N, 4, 1)
        if torch.cuda.is_available():
            ones = ones.cuda()
        xy1 = torch.cat((src_ps, ones), 2)
        zeros = torch.zeros_like(xy1)
        if torch.cuda.is_available():
            zeros = zeros.cuda()

        xyu, xyd = torch.cat((xy1, zeros), 2), torch.cat((zeros, xy1), 2)
        M1 = torch.cat((xyu, xyd), 2).reshape(N, -1, 6)
        M2 = torch.matmul(
            dst_p.reshape(-1, 2, 1),
            src_ps.reshape(-1, 1, 2),
        ).reshape(N, -1, 2)

        A = torch.cat((M1, -M2), 2)
        b = dst_p.reshape(N, -1, 1)

        Ainv = torch.inverse(A)
        h8 = torch.matmul(Ainv, b).reshape(N, 8)

        H = torch.cat((h8, ones[:, 0, :]), 1).reshape(N, 3, 3)
        H = H.reshape(bs, n, 3, 3)
        return H

    def transform(self, patch_size, M_tile_inv, H_mat, M_tile, I, patch_indices, batch_indices_tensor):
        # Transform H_mat since we scale image indices in transformer
        batch_size, num_channels, img_h, img_w = I.size()
        # if torch.cuda.is_available():
        #     M_tile_inv = M_tile_inv.cuda()
        H_mat = torch.matmul(torch.matmul(M_tile_inv, H_mat), M_tile)
        # Transform image 1 (large image) to image 2
        out_size = (img_h, img_w)
        warped_images, _ = transformer(I, H_mat, out_size)

        # Extract the warped patch from warped_images by flatting the whole batch before using indices
        # Note that input I  is  3 channels so we reduce to gray
        warped_gray_images = torch.mean(warped_images, dim=3)
        warped_images_flat = torch.reshape(warped_gray_images, [-1])
        patch_indices_flat = torch.reshape(patch_indices, [-1])
        pixel_indices = patch_indices_flat.long() + batch_indices_tensor
        pred_I2_flat = torch.gather(warped_images_flat, 0, pixel_indices)

        pred_I2 = torch.reshape(pred_I2_flat, [batch_size, patch_size, patch_size, 1])

        return pred_I2.permute(0, 3, 1, 2)

    def build_losses(self, pred_I2, I2_aug):
        rec_loss = torch.sqrt(torch.mean((pred_I2 - I2_aug) ** 2))
        ssim_loss = torch.mean(SSIM_loss(pred_I2, I2_aug))
        l1_loss = torch.mean(torch.abs(pred_I2 - I2_aug))
        l1_smooth_loss = L1_smooth_loss(pred_I2, I2_aug)
        ncc_loss = NCC_loss(I2_aug, pred_I2)
        return rec_loss, ssim_loss, l1_loss, l1_smooth_loss, ncc_loss
