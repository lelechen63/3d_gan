import cv2
import numpy as np
import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('--face_img', type=str, required=True)
parser.add_argument('--lip_folder', type=str, required=True)
parser.add_argument('--resize', nargs='+', type=int, help='e.g.: --resize 66 66')
parser.add_argument('--position', nargs='+', type=int, help='e.g.: --position 64 93')
parser.add_argument('--prefix', type=str, required=True, help='real or fake')
parser.add_argument('--output_dir', type=str, required=True)

args = parser.parse_args()
assert args.face_img is not None
assert args.output_dir is not None


resize = tuple(args.resize)
position = tuple(args.position)

for i in range(0, 938):
    face = cv2.imread(args.face_img)
    basename = args.prefix + '_{}.png'.format(i)
    lip_path = os.path.join(args.lip_folder, basename)
    print(lip_path)
    assert os.path.exists(lip_path)
    lip = cv2.imread(lip_path)
    lip = cv2.resize(lip, resize)
    mask = 255 * np.ones((resize[1], resize[0], 3), lip.dtype)
    output = cv2.seamlessClone(lip, face, mask, position, flags=cv2.MIXED_CLONE)
    cv2.imwrite(os.path.join(args.output_dir, basename), output)
