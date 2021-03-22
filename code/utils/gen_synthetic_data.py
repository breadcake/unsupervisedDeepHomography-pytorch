import os, shutil, argparse
import glob
import cv2
import random
import numpy as np
from numpy.linalg import inv
from numpy_spatial_transformer import numpy_transformer


def homographyGeneration(args, raw_image_path, index, I_dir, I_prime_dir, gt_file, pts1_file, filenames_file):
    rho = args.rho
    patch_size = args.patch_size
    height = args.img_h
    width = args.img_w

    try:
        color_image = cv2.imread(raw_image_path)
        color_image = cv2.resize(color_image, (width, height))
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    except:
        print('Error with image:', raw_image_path)
        return index, -1

    # Randomly pick the top left point of the patch on the real image
    y = random.randint(rho, height - rho - patch_size)  # row?
    x = random.randint(rho, width - rho - patch_size)  # col?

    # define corners of image patch
    top_left_point = (x, y)
    bottom_left_point = (patch_size + x, y)
    bottom_right_point = (patch_size + x, patch_size + y)
    top_right_point = (x, patch_size + y)
    four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
    perturbed_four_points = []
    for point in four_points:
        perturbed_four_points.append((point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

    # compute Homography
    H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
    try:
        H_inverse = inv(H)
    except:
        print("singular Error!")
        return index, -1

    inv_warped_color_image = None
    inv_warped_image = None
    if args.color:
        inv_warped_color_image = numpy_transformer(color_image, H_inverse, (width, height))
    else:
        inv_warped_image = numpy_transformer(gray_image, H_inverse, (width, height))

    # Extreact image patches (not used)
    if args.color:
        original_patch = gray_image[y:y + patch_size, x:x + patch_size]
    else:
        warped_patch = inv_warped_image[y:y + patch_size, x:x + patch_size]

    ######################################################################################
    # Save synthetic data I_dir I_prime_dir gt pts1
    large_img_path = os.path.join(I_dir, str(index) + '.jpg')
    if args.mode == 'train' and args.color == False:
        cv2.imwrite(large_img_path, gray_image)
    else:
        cv2.imwrite(large_img_path, color_image)

    if I_prime_dir is not None:
        img_prime_path = os.path.join(I_prime_dir, str(index) + '.jpg')
        if args.mode == 'train' and args.color == False:
            cv2.imwrite(img_prime_path, inv_warped_image)
        else:
            cv2.imwrite(img_prime_path, inv_warped_color_image)

    # Text files to store homography parameters (4 corners)
    f_pts1 = open(pts1_file, 'ab')
    f_gt = open(gt_file, 'ab')
    f_file_list = open(filenames_file, 'ab')

    # Ground truth is delta displacement
    gt = np.subtract(np.array(perturbed_four_points), np.array(four_points))
    gt = np.array(gt).flatten().astype(np.float32)
    # Four corners in the first image
    pts1 = np.array(four_points).flatten().astype(np.float32)

    np.savetxt(f_gt, [gt], fmt='%.1f', delimiter=' ')
    np.savetxt(f_pts1, [pts1], fmt='%.1f', delimiter=' ')
    f_file_list.write(('%s %s\n' % (str(index) + '.jpg', str(index) + '.jpg')).encode())

    index += 1
    if index % 1000 == 0:
        print('--image number ', index)

    f_gt.close()
    f_pts1.close()
    f_file_list.close()
    return index, 0


def dataCollection(args):
    # Default folders and files for storage
    DATA_PATH = None
    gt_file = None
    pts1_file = None
    filenames_file = None
    if args.mode == 'train':
        DATA_PATH = args.data_path + 'train/'
        pts1_file = os.path.join(DATA_PATH, 'pts1.txt')
        filenames_file = os.path.join(DATA_PATH, 'train_synthetic.txt')
        gt_file = os.path.join(DATA_PATH, 'gt.txt')
    elif args.mode == 'test':
        DATA_PATH = args.data_path + 'test/'
        pts1_file = os.path.join(DATA_PATH, 'test_pts1.txt')
        filenames_file = os.path.join(DATA_PATH, 'test_synthetic.txt')
        gt_file = os.path.join(DATA_PATH, 'test_gt.txt')
    I_dir = DATA_PATH + 'I/'  # Large image
    I_prime_dir = DATA_PATH + 'I_prime/'  # Large image

    try:
        os.remove(gt_file)
        os.remove(pts1_file)
        os.remove(filenames_file)
        print('-- Current {} existed. Deleting..!'.format(gt_file))
        shutil.rmtree(I_dir, ignore_errors=True)
        if I_prime_dir is not None:
            shutil.rmtree(I_prime_dir, ignore_errors=True)
    except:
        print('-- Current {} not existed yet!'.format(gt_file))

    if not os.path.exists(I_dir):
        os.makedirs(I_dir)
    if I_prime_dir is not None and not os.path.exists(I_prime_dir):
        os.makedirs(I_prime_dir)

    raw_image_list = glob.glob(os.path.join(args.raw_data_path, '*.jpg'))
    print("Generate {} {} files from {} raw data.".format(args.num_data, args.mode, len(raw_image_list)))

    index = 0
    while True:
        raw_img_name = random.choice(raw_image_list)
        raw_image_path = os.path.join(args.raw_data_path, raw_img_name)
        index, error = homographyGeneration(args, raw_image_path, index,
                                            I_dir, I_prime_dir, gt_file, pts1_file, filenames_file)
        if error == -1:
            continue
        if index >= args.num_data:
            break


def main():
    RHO = 45  # The maximum value of pertubation

    DATA_NUMBER = 100000
    TEST_DATA_NUMBER = 5000

    # Size of synthetic image
    HEIGHT = 240  #
    WIDTH = 320
    PATCH_SIZE = 128

    # Directories to files
    RAW_DATA_PATH = "D:/Workspace/Datasets/coco2014/train2014/"  # Real images used for generating synthetic data
    TEST_RAW_DATA_PATH = "D:/Workspace/Datasets/coco2014/test2014/"  # Real images used for generating test synthetic data

    # Synthetic data directories
    DATA_PATH = "../data/synthetic/" + str(RHO) + '/'
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    def str2bool(s):
        return s.lower() == 'true'

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', help='Train or test', choices=['train', 'test'])
    parser.add_argument('--color', type=str2bool, default='true', help='Generate color or gray images')

    parser.add_argument('--raw_data_path', type=str, default=RAW_DATA_PATH, help='The raw data path.')
    parser.add_argument('--test_raw_data_path', type=str, default=TEST_RAW_DATA_PATH, help='The test raw data path.')
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help='The raw data path.')
    parser.add_argument('--num_data', type=int, default=DATA_NUMBER, help='The data size for training')
    parser.add_argument('--test_num_data', type=int, default=TEST_DATA_NUMBER, help='The data size for test')

    parser.add_argument('--img_w', type=int, default=WIDTH)
    parser.add_argument('--img_h', type=int, default=HEIGHT)
    parser.add_argument('--rho', type=int, default=RHO)
    parser.add_argument('--patch_size', type=int, default=PATCH_SIZE)

    args = parser.parse_args()
    print('<==================== Loading raw data ===================>\n')
    if args.mode == 'test':
        args.num_data = args.test_num_data
        args.raw_data_path = args.test_raw_data_path

    print('<================= Generating Data .... =================>\n')

    dataCollection(args)


if __name__ == '__main__':
    main()
