import cv2
import numpy as np
import argparse
import os
import re
from tqdm import tqdm


# def filter(image):
#     img = cv2.imread(image)
#     count = 0
#     for i in range(480):
#         for j in range(640):
#             if img[i][j][0] == 255 and img[i][j][1] == 255 and img[i][j][2] == 255:
#                 count += 1
#     rate = count/(640*480)
#     dirs = image.split('/')
#     date = dirs[4]
#     month = date[:6]
#     if rate <= 0.4:
#         p = os.path.join(dst_root, month)
#         if not os.path.isdir(p):
#             os.mkdir(p)
#         cv2.imwrite(p+'/'+dirs[-1], img)
#     else:
#         white.append(image)


def filter(image):
    img = cv2.imread(image)
    white = np.array([255, 255, 255], np.uint8)
    mask = cv2.inRange(img, white, white)
    count = cv2.countNonZero(mask)
    rate = count / (640*480)
    if rate <= 0.4:
        p = '/data_1/washington-square/front2/'
        cv2.imwrite(p+image.split('/')[-1], img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    args = parser.parse_args()

    # if not os.path.isdir('/data_1/washington-square/front2/'):
    #     os.system('mkdir /data_1/washington-square/front2/')

    root_dir = args.image_path
    images = []
    for (root, dirs, files) in os.walk(root_dir, topdown=True):
        if len(files) != 0:
            images += [os.path.join(root, f) for f in files]
    
    images.sort()
    for image in tqdm(images):
        rate = filter(image)
