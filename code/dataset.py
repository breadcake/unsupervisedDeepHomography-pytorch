import os
import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset

def read_img_and_gt(filenames_file, pts1_file, gt_file):
    with open(filenames_file, 'r') as img_f:
        img_array = img_f.readlines()
    img_array = [x.strip() for x in img_array]
    img_array = [x.split() for x in img_array]  # Use x.split()[0] if assuming image left and right have same name

    with open(pts1_file, 'r') as pts1_f:
        pts1_array = pts1_f.readlines()
    pts1_array = [x.strip() for x in pts1_array]
    pts1_array = [x.split() for x in pts1_array]
    pts1_array = np.array(pts1_array).astype('float64')

    # In case there is not ground truth
    if not gt_file:
        return img_array, pts1_array, None

    with open(gt_file, 'r') as gt_f:
        gt_array = gt_f.readlines()
    gt_array = [x.strip() for x in gt_array]
    gt_array = [x.split() for x in gt_array]
    gt_array = np.array(gt_array).astype('float64')

    return img_array, pts1_array, gt_array

def get_mesh_grid_per_img(patch_w, patch_h):
    x_flat = np.arange(0, patch_w)
    x_flat = x_flat[np.newaxis, :]
    y_one = np.ones(patch_h)
    y_one = y_one[:, np.newaxis]
    x_mesh = np.matmul(y_one, x_flat)

    y_flat = np.arange(0, patch_h)
    y_flat = y_flat[:, np.newaxis]
    x_one = np.ones(patch_w)
    x_one = x_one[np.newaxis, :]
    y_mesh = np.matmul(y_flat, x_one)
    x_t_flat = np.reshape(x_mesh, (-1))
    y_t_flat = np.reshape(y_mesh, (-1))
    return x_t_flat, y_t_flat

class SyntheticDataset(Dataset):
    """Load synthetic data"""
    def __init__(self, data_path, mode, img_h, img_w, patch_size, do_augment):
        if mode == "train":
            self.data_path = data_path + "train/"
            self.pts1_file = os.path.join(self.data_path, 'pts1.txt')
            self.filenames_file = os.path.join(self.data_path, 'train_synthetic.txt')
            self.gt_file = os.path.join(self.data_path, 'gt.txt')
        elif mode == "test":
            self.data_path = data_path + "test/"
            self.pts1_file = os.path.join(self.data_path, 'test_pts1.txt')
            self.filenames_file = os.path.join(self.data_path, 'test_synthetic.txt')
            self.gt_file = os.path.join(self.data_path, 'test_gt.txt')
        self.img_h = img_h
        self.img_w = img_w
        self.patch_size = patch_size
        self.do_augment = do_augment
        self.mean_I = np.reshape(np.array([118.93, 113.97, 102.60]), (1, 1, 3))
        self.std_I = np.reshape(np.array([69.85, 68.81, 72.45]), (1, 1, 3))

        # Read to arrays
        self.img_np, self.pts1_np, self.gt_np = read_img_and_gt(self.filenames_file, self.pts1_file, self.gt_file)

        # Find indices of the pixels in the patch w.r.t the large image
        # All patches have the same size so their pixels have the same base indices
        self.x_t_flat, self.y_t_flat = get_mesh_grid_per_img(patch_size, patch_size)

    def __len__(self):
        return len(self.img_np)

    def __getitem__(self, index):
        pts1_index = self.pts1_np[index]
        gt_index = self.gt_np[index]
        split_line = self.img_np[index]

        I_path = self.data_path + 'I/' + split_line[0]
        I_prime_path = self.data_path + 'I_prime/' + split_line[1]

        I = self.read_image(I_path, self.img_h, self.img_w)
        I_prime = self.read_image(I_prime_path, self.img_h, self.img_w)

        # Data Augmentation
        do_augment = np.random.uniform(0, 1)
        I_aug, I_prime_aug = self.joint_augment_image_pair(I, I_prime, 0, 255) \
            if do_augment > (1 - self.do_augment) else (I, I_prime)

        # Standardize images
        I = self.norm_img(I, self.mean_I, self.std_I)
        I_prime = self.norm_img(I_prime, self.mean_I, self.std_I)
        # These are augmented large images which will be used
        I_aug = self.norm_img(I_aug, self.mean_I, self.std_I)
        I_prime_aug = self.norm_img(I_prime_aug, self.mean_I, self.std_I)

        # Read patch_indices
        x_start = pts1_index[0]  # x
        y_start = pts1_index[1]  # y
        patch_indices = (self.y_t_flat + y_start) * self.img_w + (self.x_t_flat + x_start)

        # Convert to tensor
        I = torch.tensor(I)
        I_prime = torch.tensor(I_prime)
        I_aug = torch.tensor(I_aug)
        I_prime_aug = torch.tensor(I_prime_aug)
        pts1_tensor = torch.tensor(pts1_index)
        gt_tensor = torch.tensor(gt_index)
        patch_indices = torch.tensor(patch_indices)

        # Obtain I1, I2, I1_aug and I2_aug
        I_flat = torch.reshape(torch.mean(I, 0), [-1])  # I: 3xHxW
        I_prime_flat = torch.reshape(torch.mean(I_prime, 0), [-1])  # I_prime: 3xHxW
        I_aug_flat = torch.reshape(torch.mean(I_aug, 0), [-1])  # I_aug: 3xHxW
        I_prime_aug_flat = torch.reshape(torch.mean(I_prime_aug, 0), [-1])  # I_prime_aug: 3xHxW

        patch_indices_flat = torch.reshape(patch_indices, [-1])
        pixel_indices = patch_indices_flat.long()

        I1_flat = torch.gather(I_flat, 0, pixel_indices)
        I2_flat = torch.gather(I_prime_flat, 0, pixel_indices)
        I1_aug_flat = torch.gather(I_aug_flat, 0, pixel_indices)
        I2_aug_flat = torch.gather(I_prime_aug_flat, 0, pixel_indices)

        I1 = torch.reshape(I1_flat, [self.patch_size, self.patch_size, 1]).permute(2, 0, 1)
        I2 = torch.reshape(I2_flat, [self.patch_size, self.patch_size, 1]).permute(2, 0, 1)
        I1_aug = torch.reshape(I1_aug_flat, [self.patch_size, self.patch_size, 1]).permute(2, 0, 1)
        I2_aug = torch.reshape(I2_aug_flat, [self.patch_size, self.patch_size, 1]).permute(2, 0, 1)
        # import matplotlib.pyplot as plt
        # plt.subplot(221), plt.imshow(self.denorm_img(I.numpy(), self.mean_I, self.std_I)[:,:,::-1])
        # plt.subplot(222), plt.imshow(self.denorm_img(I_prime.numpy(), self.mean_I, self.std_I)[:,:,::-1])
        # plt.subplot(223), plt.imshow(self.denorm_img(I_aug.numpy(), self.mean_I, self.std_I)[:,:,::-1])
        # plt.subplot(224), plt.imshow(self.denorm_img(I_prime_aug.numpy(), self.mean_I, self.std_I)[:,:,::-1])
        #
        # plt.figure()
        # plt.gray()
        # plt.subplot(221), plt.imshow(I1.permute(1,2,0))
        # plt.subplot(222), plt.imshow(I2.permute(1,2,0))
        # plt.subplot(223), plt.imshow(I1_aug.permute(1,2,0))
        # plt.subplot(224), plt.imshow(I2_aug.permute(1,2,0))
        # plt.show()
        return I1, I2, I1_aug, I2_aug, I_aug, I_prime_aug, pts1_tensor, gt_tensor, patch_indices

    def read_image(self, image_path, img_h, img_w):
        image = cv.imread(image_path)
        height, width = image.shape[:2]
        if height != img_h or width != img_w:
            image = cv.resize(image, (img_w, img_h), interpolation=cv.INTER_AREA)
        return image

    def norm_img(self, img, mean, std):
        img = (img - mean) / std
        img = np.transpose(img, [2, 0, 1])  # torch [C,H,W]
        return img

    def denorm_img(self, img, mean, std):
        img = np.transpose(img, [1, 2, 0])
        img = img * std + mean
        return np.uint8(img)

    def joint_augment_image_pair(self, img1, img2, min_val=0, max_val=255):
        # Randomly shift gamma
        random_gamma = np.random.uniform(0.8, 1.2)
        img1_aug = img1 ** random_gamma
        img2_aug = img2 ** random_gamma

        # Randomly shift brightness
        random_brightness = np.random.uniform(0.5, 2.0)
        img1_aug = img1_aug * random_brightness
        img2_aug = img2_aug * random_brightness

        # Randomly shift color
        random_colors = np.random.uniform(0.8, 1.2, 3)
        white = np.ones([img1.shape[0], img1.shape[1], 1])
        color_image = np.concatenate([white * random_colors[i] for i in range(3)], axis=2)
        img1_aug *= color_image
        img2_aug *= color_image

        # Saturate
        img1_aug = np.clip(img1_aug, min_val, max_val)
        img2_aug = np.clip(img2_aug, min_val, max_val)

        return img1_aug, img2_aug


if __name__ == "__main__":
    TrainDataset = SyntheticDataset(data_path="../data/synthetic/45/train/",
                                    img_h=240,
                                    img_w=320,
                                    patch_size=128,
                                    do_augment=0.5)
    print(len(TrainDataset))
    sample = TrainDataset[np.random.randint(0, len(TrainDataset))]
    for attr in sample:
        print(attr.shape, attr.dtype)