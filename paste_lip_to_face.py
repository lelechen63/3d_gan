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

face = cv2.imread(args.face_img)

lip_files = glob.glob(os.path.join(args.lip_folder, args.prefix + '_*.png'))
resize = tuple(args.resize)
position = tuple(args.position)
mask = 255 * np.ones((resize[1], resize[0]), np.uint8)

assert len(lip_files) > 0

for lip_f in lip_files:
    lip = cv2.imread(lip_f)
    basename = os.path.basename(lip_f)
    lip = cv2.resize(lip, resize)
    output = cv2.seamlessClone(lip, face, mask, position, cv2.MIXED_CLONE)
    cv2.imwrite(os.path.join(args.output_dir, basename), output)
