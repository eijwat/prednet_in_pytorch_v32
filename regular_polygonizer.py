import sys
import numpy as np
import cv2
import matplotlib.colors as mcolors

import argparse

usage = 'Usage: python {} INPUT_FILE [--help]'.format(__file__)
parser = argparse.ArgumentParser(description='This script is to polygonize images from a single image.',
                                 usage=usage)
parser.add_argument('input_image', action='store', nargs=None, 
                    type=str, help='Input image.')
parser.add_argument('--size', '-s', default='160,120',
                    help='Size of resized images. width,height (pixels)')
parser.add_argument('--nside', '-n', type=int, default=4, help='N-side.')
parser.add_argument('--ratio', '-r', type=float, default=0.6, help='Ratio.')
parser.add_argument('--backgound', '-b', type=str, default='#000000', help='Background color code.')
parser.add_argument('--output', '-o', type=str, default='output.png', help='Output image filename.')
args = parser.parse_args()
if args.nside < 4:
    print("nside must be greater than 3.")
    sys.exit()

size = args.size.split(',')
size = (int(size[1]), int(size[0]))
img = cv2.imread(args.input_image)
img = cv2.resize(img, size)
color_bgr = (np.array(mcolors.to_rgb(args.backgound)) * 255).astype(np.uint8)[::-1]

drad = (2.0 * np.pi) / args.nside
r = 0.5 * size[0] / np.sin(0.5 * drad)
out_height = np.ceil(r * np.cos(0.5 * drad) * 2)
out_width = out_height if args.nside % 4 == 0 else np.ceil(r * 2)
h3 = np.ceil(r * args.ratio)
new_height = np.ceil(h3 * np.cos(0.5 * drad))
width_offset = h3 * np.sin(0.5 * drad)
p_original = np.float32([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]])
center = out_width / 2.0
ext = size[0] / 2.0
p_trans = np.float32(
    [
        [center - ext, 0],
        [center + ext, 0],
        [center + ext - width_offset, new_height],
        [center - ext + width_offset, new_height],
    ]
)
M = cv2.getPerspectiveTransform(p_original, p_trans)
out_piece = cv2.warpPerspective(img, M, (int(out_width), int(out_height)))

out_img = np.zeros((int(out_height), int(out_width), 3), dtype=np.uint8)
for i in range(args.nside):
    trans = cv2.getRotationMatrix2D((int(out_width) // 2, int(out_height) // 2), np.rad2deg(i * drad), 1.0)
    piece = cv2.warpAffine(out_piece, trans, (int(out_width), int(out_height)))
    out_img = cv2.add(out_img, piece)


def generate_nside_poly(nside: int, org_pt: np.ndarray) -> np.ndarray:
    pts = []
    for i in range(nside):
        th = i * drad
        cth = np.cos(th)
        sth = np.sin(th)
        rot = np.array([[cth, -sth], [sth, cth]])
        offset = np.dot(rot, org_pt)
        pts.append([out_width / 2 + offset[0], out_height / 2 - offset[1]])
    return np.array(pts, dtype=int)


# inner
org_pt = np.array([r * np.sin(drad * 0.5), r * np.cos(drad * 0.5)]) * (1.0 - args.ratio)
inner_pts = generate_nside_poly(args.nside, org_pt)
mask = np.zeros((int(out_height), int(out_width), 3), dtype=np.uint8)
mask = cv2.fillConvexPoly(mask, inner_pts, color=color_bgr.tolist())
out_img = cv2.add(out_img, mask)

# outer
org_pt = np.array([r * np.sin(drad * 0.5), r * np.cos(drad * 0.5)])
outer_pts = generate_nside_poly(args.nside, org_pt)
mask = np.tile(color_bgr, (int(out_height), int(out_width), 1)).astype(np.uint8)
mask = cv2.fillConvexPoly(mask, outer_pts, color=(0, 0, 0))
out_img = cv2.add(out_img, mask)

cv2.imwrite(args.output, out_img)
