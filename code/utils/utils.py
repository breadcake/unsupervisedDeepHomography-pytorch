import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def denorm_img(img):
    img = np.transpose(img, [1, 2, 0])
    if len(img.shape) == 3:
        mean = np.array([118.93, 113.97, 102.60]).reshape([1, 1, 3])
        std = np.array([69.85, 68.81, 72.45]).reshape([1, 1, 3])
    elif len(img.shape) == 2:
        mean = np.mean([118.93, 113.97, 102.60])
        std = np.mean([69.85, 68.81, 72.45])
    return img * std + mean


def save_correspondences_img(img1, img2, corr1, corr2, pred_corr2, results_dir, img_name):
    """ Save pair of images with their correspondences into a single image. Used for report"""
    # Draw prediction
    copy_img2 = img2.copy()
    copy_img1 = img1.copy()
    cv2.polylines(copy_img2, np.int32([pred_corr2]), 1, (5, 225, 225), 3)

    point_color = (0, 255, 255)
    line_color_set = [(255, 102, 255), (51, 153, 255), (102, 255, 255), (255, 255, 0), (102, 102, 244), (150, 202, 178),
                      (153, 240, 142), (102, 0, 51), (51, 51, 0)]
    # Draw 4 points (ground truth)
    full_stack_images = draw_matches(copy_img1, corr1, copy_img2, corr2, '/tmp/tmp.jpg', color_set=line_color_set,
                                     show=False)
    # Save image
    visual_file_name = os.path.join(results_dir, img_name)
    # cv2.putText(full_stack_images, 'RMSE %.2f'%h_loss,(800, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
    cv2.imwrite(visual_file_name, full_stack_images)
    print('Wrote file %s', visual_file_name)


def draw_matches(img1, kp1, img2, kp2, output_img_file=None, color_set=None, show=True):
    """Draws lines between matching keypoints of two images without matches.
    This is a replacement for cv2.drawMatches
    Places the images side by side in a new image and draws circles
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        color_set: The colors of the circles and connecting lines drawn on the images.
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2

    # Draw lines between points

    kp2_on_stack_image = (kp2 + np.array([img1.shape[1], 0])).astype(np.int32)

    kp1 = kp1.astype(np.int32)
    # kp2_on_stack_image[0:4,0:2]
    line_color1 = (2, 10, 240)
    line_color2 = (2, 10, 240)
    # We want to make connections between points to make a square grid so first count the number of rows in the square grid.
    grid_num_rows = int(np.sqrt(kp1.shape[0]))

    if output_img_file is not None and grid_num_rows >= 3:
        for i in range(grid_num_rows):
            cv2.line(new_img, tuple(kp1[i*grid_num_rows]), tuple(kp1[i*grid_num_rows + (grid_num_rows-1)]), line_color1, 1,  LINE_AA)
            cv2.line(new_img, tuple(kp1[i]), tuple(kp1[i + (grid_num_rows-1)*grid_num_rows]), line_color1, 1,  cv2.LINE_AA)
            cv2.line(new_img, tuple(kp2_on_stack_image[i*grid_num_rows]), tuple(kp2_on_stack_image[i*grid_num_rows + (grid_num_rows-1)]), line_color2, 1,  cv2.LINE_AA)
            cv2.line(new_img, tuple(kp2_on_stack_image[i]), tuple(kp2_on_stack_image[i + (grid_num_rows-1)*grid_num_rows]), line_color2, 1,  cv2.LINE_AA)

    if output_img_file is not None and grid_num_rows == 2:
        cv2.polylines(new_img, np.int32([kp2_on_stack_image]), 1, line_color2, 3)
        cv2.polylines(new_img, np.int32([kp1]), 1, line_color1, 3)
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 7
    thickness = 1

    for i in range(len(kp1)):
        key1 = kp1[i]
        key2 = kp2[i]
        # Generate random color for RGB/BGR and grayscale images as needed.
        try:
            c  = color_set[i]
        except:
            c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(key1).astype(int))
        end2 = tuple(np.round(key2).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness,  cv2.LINE_AA)
        cv2.circle(new_img, end1, r, c, thickness,  cv2.LINE_AA)
        cv2.circle(new_img, end2, r, c, thickness,  cv2.LINE_AA)
    # pdb.set_trace()
    if show:
        plt.figure(figsize=(15,15))
        if len(img1.shape) == 3:
            plt.imshow(new_img)
        else:
            plt.imshow(new_img)
        plt.axis('off')
        plt.show()
    if output_img_file is not None:
        cv2.imwrite(output_img_file, new_img)

    return new_img