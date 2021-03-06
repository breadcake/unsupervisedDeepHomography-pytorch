import os, shutil
import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dataset import SyntheticDataset
from homography_model import HomographyModel
from utils import utils
import numpy as np
import math
import time

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def train(args):
    # Load data
    TrainDataset = SyntheticDataset(data_path=args.data_path,
                                    mode=args.mode,
                                    img_h=args.img_h,
                                    img_w=args.img_w,
                                    patch_size=args.patch_size,
                                    do_augment=args.do_augment)
    train_loader = DataLoader(TrainDataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print('===> Train: There are totally {} training files'.format(len(TrainDataset)))

    net = HomographyModel(args.use_batch_norm)
    if args.resume:
        model_path = os.path.join(args.model_dir, args.model_name)
        ckpt = torch.load(model_path)
        net.load_state_dict(ckpt.state_dict())
    if torch.cuda.is_available():
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=args.lr)  # default as 0.0001
    decay_rate = 0.96
    step_size = (math.log(decay_rate) * args.max_epochs) / math.log(args.min_lr * 1.0 / args.lr)
    print('args lr:', args.lr, args.min_lr)
    print('===> Decay steps:', step_size)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(step_size), gamma=0.96)

    print("start training")
    writer = SummaryWriter(logdir=args.log_dir, flush_secs=60)
    score_print_fre = 100
    summary_fre = 1000
    model_save_fre = 4000
    glob_iter = 0
    t0 = time.time()

    for epoch in range(args.max_epochs):
        net.train()
        epoch_start = time.time()
        train_l1_loss = 0.0
        train_l1_smooth_loss = 0.0
        train_h_loss = 0.0

        for i, batch_value in enumerate(train_loader):
            I1_batch = batch_value[0].float()
            I2_batch = batch_value[1].float()
            I1_aug_batch = batch_value[2].float()
            I2_aug_batch = batch_value[3].float()
            I_batch = batch_value[4].float()
            I_prime_batch = batch_value[5].float()
            pts1_batch = batch_value[6].float()
            gt_batch = batch_value[7].float()
            patch_indices_batch = batch_value[8].float()

            if torch.cuda.is_available():
                I1_aug_batch = I1_aug_batch.cuda()
                I2_aug_batch = I2_aug_batch.cuda()
                I_batch = I_batch.cuda()
                pts1_batch = pts1_batch.cuda()
                gt_batch = gt_batch.cuda()
                patch_indices_batch = patch_indices_batch.cuda()

            # forward, backward, update weights
            optimizer.zero_grad()
            batch_out = net(I1_aug_batch, I2_aug_batch, I_batch, pts1_batch, gt_batch, patch_indices_batch)
            h_loss = batch_out['h_loss']
            rec_loss = batch_out['rec_loss']
            ssim_loss = batch_out['ssim_loss']
            l1_loss = batch_out['l1_loss']
            l1_smooth_loss = batch_out['l1_smooth_loss']
            ncc_loss = batch_out['ncc_loss']
            pred_I2 = batch_out['pred_I2']

            loss = l1_loss
            loss.backward()
            optimizer.step()

            train_l1_loss += loss.item()
            train_l1_smooth_loss += l1_smooth_loss.item()
            train_h_loss += h_loss.item()
            if (i + 1) % score_print_fre == 0 or (i + 1) == len(train_loader):
                print(
                    "Training: Epoch[{:0>3}/{:0>3}] Iter[{:0>3}]/[{:0>3}] l1 loss: {:.4f} "
                    "l1 smooth loss: {:.4f} h loss: {:.4f} lr={:.8f}".format(
                        epoch + 1, args.max_epochs, i + 1, len(train_loader), train_l1_loss / score_print_fre,
                        train_l1_smooth_loss / score_print_fre, train_h_loss / score_print_fre, scheduler.get_lr()[0]))
                train_l1_loss = 0.0
                train_l1_smooth_loss = 0.0
                train_h_loss = 0.0

            if glob_iter % summary_fre == 0:
                writer.add_scalar('learning_rate', scheduler.get_lr()[0], glob_iter)
                writer.add_scalar('h_loss', h_loss, glob_iter)
                writer.add_scalar('rec_loss', rec_loss, glob_iter)
                writer.add_scalar('ssim_loss', ssim_loss, glob_iter)
                writer.add_scalar('l1_loss', l1_loss, glob_iter)
                writer.add_scalar('l1_smooth_loss', l1_smooth_loss, glob_iter)
                writer.add_scalar('ncc_loss', ncc_loss, glob_iter)

                writer.add_image('I', utils.denorm_img(I_batch[0, ...].cpu().numpy()).astype(np.uint8)[:, :, ::-1],
                                 glob_iter, dataformats='HWC')
                writer.add_image('I_prime',
                                 utils.denorm_img(I_prime_batch[0, ...].numpy()).astype(np.uint8)[:, :, ::-1],
                                 glob_iter, dataformats='HWC')

                writer.add_image('I1_aug', utils.denorm_img(I1_aug_batch[0, 0, ...].cpu().numpy()).astype(np.uint8),
                                 glob_iter, dataformats='HW')
                writer.add_image('I2_aug', utils.denorm_img(I2_aug_batch[0, 0, ...].cpu().numpy()).astype(np.uint8),
                                 glob_iter, dataformats='HW')
                writer.add_image('pred_I2',
                                 utils.denorm_img(pred_I2[0, 0, ...].cpu().detach().numpy()).astype(np.uint8),
                                 glob_iter, dataformats='HW')

                writer.add_image('I2', utils.denorm_img(I2_batch[0, 0, ...].numpy()).astype(np.uint8), glob_iter,
                                 dataformats='HW')
                writer.add_image('I1', utils.denorm_img(I1_batch[0, 0, ...].numpy()).astype(np.uint8), glob_iter,
                                 dataformats='HW')

            # save model
            if glob_iter % model_save_fre == 0 and glob_iter != 0:
                filename = 'model' + '_iter_' + str(glob_iter) + '.pth'
                model_save_path = os.path.join(args.model_dir, filename)
                torch.save(net, model_save_path)

            glob_iter += 1
        scheduler.step()
        print("Epoch: {} epoch time: {:.1f}s".format(epoch, time.time() - epoch_start))

    elapsed_time = time.time() - t0
    print("Finished Training in {:.0f}h {:.0f}m {:.0f}s.".format(
        elapsed_time // 3600, (elapsed_time % 3600) // 60, (elapsed_time % 3600) % 60))


def test(args):
    # Load data
    TestDataset = SyntheticDataset(data_path=args.data_path,
                                   mode=args.mode,
                                   img_h=args.img_h,
                                   img_w=args.img_w,
                                   patch_size=args.patch_size,
                                   do_augment=args.do_augment)
    test_loader = DataLoader(TestDataset, batch_size=1)
    print('===> Test: There are totally {} testing files'.format(len(TestDataset)))

    # Load model
    net = HomographyModel()
    model_path = os.path.join(args.model_dir, args.model_name)
    state = torch.load(model_path)
    net.load_state_dict(state.state_dict())
    if torch.cuda.is_available():
        net = net.cuda()

    print("start testing")

    with torch.no_grad():
        net.eval()
        test_l1_loss = 0.0
        test_h_loss = 0.0
        h_losses_array = []
        for i, batch_value in enumerate(test_loader):
            I1_aug_batch = batch_value[2].float()
            I2_aug_batch = batch_value[3].float()
            I_batch = batch_value[4].float()
            I_prime_batch = batch_value[5].float()
            pts1_batch = batch_value[6].float()
            gt_batch = batch_value[7].float()
            patch_indices_batch = batch_value[8].float()

            if torch.cuda.is_available():
                I1_aug_batch = I1_aug_batch.cuda()
                I2_aug_batch = I2_aug_batch.cuda()
                I_batch = I_batch.cuda()
                pts1_batch = pts1_batch.cuda()
                gt_batch = gt_batch.cuda()
                patch_indices_batch = patch_indices_batch.cuda()

            batch_out = net(I1_aug_batch, I2_aug_batch, I_batch, pts1_batch, gt_batch, patch_indices_batch)
            h_loss = batch_out['h_loss']
            rec_loss = batch_out['rec_loss']
            ssim_loss = batch_out['ssim_loss']
            l1_loss = batch_out['l1_loss']
            pred_h4p_value = batch_out['pred_h4p']

            test_h_loss += h_loss.item()
            test_l1_loss += l1_loss.item()
            h_losses_array.append(h_loss.item())

            if args.save_visual:
                I_sample = utils.denorm_img(I_batch[0].cpu().numpy()).astype(np.uint8)
                I_prime_sample = utils.denorm_img(I_prime_batch[0].numpy()).astype(np.uint8)
                pts1_sample = pts1_batch[0].cpu().numpy().reshape([4, 2]).astype(np.float32)
                gt_h4p_sample = gt_batch[0].cpu().numpy().reshape([4, 2]).astype(np.float32)

                pts2_sample = pts1_sample + gt_h4p_sample

                pred_h4p_sample = pred_h4p_value[0].cpu().numpy().reshape([4, 2]).astype(np.float32)
                pred_pts2_sample = pts1_sample + pred_h4p_sample

                # Save
                visual_file_name = ('%s' % i).zfill(4) + '.jpg'
                utils.save_correspondences_img(I_prime_sample, I_sample, pts1_sample, pts2_sample, pred_pts2_sample,
                                               args.results_dir, visual_file_name)

            print("Testing: h_loss: {:4.3f}, rec_loss: {:4.3f}, ssim_loss: {:4.3f}, l1_loss: {:4.3f}".format(
                h_loss.item(), rec_loss.item(), ssim_loss.item(), l1_loss.item()
            ))

    print('|Test size  |   h_loss   |    l1_loss   |')
    print(len(test_loader), test_h_loss / len(test_loader), test_l1_loss / len(test_loader))

    tops_list = utils.find_percentile(h_losses_array)
    print('===> Percentile Values: (20, 50, 80, 100):')
    print(tops_list)
    print('======> End! ====================================')


def main():
    # Size of synthetic image and the pertubation range (RH0)
    HEIGHT = 240  #
    WIDTH = 320
    RHO = 45
    PATCH_SIZE = 128

    # Synthetic data directories
    DATA_PATH = "../data/synthetic/" + str(RHO) + '/'

    # Log and model directories
    MAIN_LOG_PATH = '../'
    LOG_DIR = MAIN_LOG_PATH + "logs/"
    MODEL_DIR = MAIN_LOG_PATH + "models/synthetic_models"

    # Where to save visualization images (for report)
    RESULTS_DIR = MAIN_LOG_PATH + "results/synthetic/report/"

    def str2bool(s):
        return s.lower() == 'true'

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='Train or test', choices=['train', 'test'])
    parser.add_argument('--use_batch_norm', type=str2bool, default='False', help='Use batch_norm?')
    parser.add_argument('--do_augment', type=float, default=0.5,
                        help='Possibility of augmenting image: color shift, brightness shift...')

    parser.add_argument('--data_path', type=str, default=DATA_PATH, help='The raw data path.')
    parser.add_argument('--log_dir', type=str, default=LOG_DIR, help='The log path')
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR, help='Store visualization for report')
    parser.add_argument('--model_dir', type=str, default=MODEL_DIR, help='The models path')
    parser.add_argument('--model_name', type=str, default='model.pth', help='The model name')

    parser.add_argument('--save_visual', type=str2bool, default='True', help='Save visual images for report')

    parser.add_argument('--img_w', type=int, default=WIDTH)
    parser.add_argument('--img_h', type=int, default=HEIGHT)
    parser.add_argument('--patch_size', type=int, default=PATCH_SIZE)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-4, help='Max learning rate')
    parser.add_argument('--min_lr', type=float, default=.9e-4, help='Min learning rate')

    parser.add_argument('--resume', type=str2bool, default='False',
                        help='True: restore the existing model. False: retrain')

    args = parser.parse_args()
    print('<==================== Loading data ===================>\n')

    if not args.resume:
        try:
            shutil.rmtree(args.log_dir)
        except:
            pass

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    if args.mode == "train":
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
