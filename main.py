import os
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import sam_model_registry

from automatic_mask_and_probability_generator import \
    SamAutomaticMaskAndProbabilityGenerator

import argparse

def normalize_image(image):
    # Normalize the image to the range [0, 1]
    min_val = image.min()
    max_val = image.max()
    image = (image - min_val) / (max_val - min_val)

    return image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pp', type=str, default='threshold')
    return parser.parse_args()


def main():
    device = "cuda"
    sam = sam_model_registry["default"](
        checkpoint="/working/model/sam_vit_h_4b8939.pth")
    sam.to(device=device)
    generator = SamAutomaticMaskAndProbabilityGenerator(sam)

    img_path = osp.join('assets/fish.jpg')
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = generator.generate(image)

    p_max = None
    for mask in masks:
        p = mask["prob"]
        if p_max is None:
            p_max = p
        else:
            p_max = np.maximum(p_max, p)

    p_max = normalize_image(p_max)

    args = get_args()
    assert args.pp in ['threshold', 'canny']

    if args.pp == 'threshold':
        # p_max min-max normalization
        p_max[p_max < 0.5] = 0
        p_max[p_max >= 0.5] = 1
        edges = (p_max * 255).astype(np.uint8)
    elif args.pp == 'canny':
        # Canny edge detection
        p_max = (p_max * 255).astype(np.uint8)
        edges = cv2.Canny(p_max, threshold1=100, threshold2=200)

    # make output directory
    os.makedirs('output', exist_ok=True)
    plt.imsave('output/edge.png', edges, cmap='binary')


if __name__ == "__main__":
    main()
