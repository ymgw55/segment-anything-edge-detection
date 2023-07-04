import os
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import sam_model_registry

from automatic_mask_and_probability_generator import \
    SamAutomaticMaskAndProbabilityGenerator
from edge_nms import Canny, normalize_image


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

    # make output directory
    os.makedirs('output', exist_ok=True)

    # p_max min-max normalization
    p_max = normalize_image(p_max)
    plt.imsave('output/prob.png', p_max, cmap='binary')

    # Canny edge detection
    image, _ = Canny(p_max, 0, 50)
    plt.imsave('output/edge.png', image, cmap='binary')


if __name__ == "__main__":
    main()
