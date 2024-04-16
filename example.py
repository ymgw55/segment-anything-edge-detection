from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import sam_model_registry

from automatic_mask_and_probability_generator import \
    SamAutomaticMaskAndProbabilityGenerator


def normalize_image(image):
    # Normalize the image to the range [0, 1]
    min_val = image.min()
    max_val = image.max()
    image = (image - min_val) / (max_val - min_val)

    return image


def main():
    device = "cuda"
    sam = sam_model_registry["default"](
        checkpoint="model/sam_vit_h_4b8939.pth")
    sam.to(device=device)
    generator = SamAutomaticMaskAndProbabilityGenerator(sam)

    img_path = 'assets/fish.jpg'
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

    edges = normalize_image(p_max)
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(
        'model/model.yml.gz')
    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    # make output directory
    Path('output/example').mkdir(parents=True, exist_ok=True)
    plt.imsave('output/example/edge.png', edges, cmap='binary')


if __name__ == "__main__":
    main()
