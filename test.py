from pathlib import Path

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import sam_model_registry
from tqdm import tqdm

from automatic_mask_and_probability_generator import \
    SamAutomaticMaskAndProbabilityGenerator


def normalize_image(image):
    # Normalize the image to the range [0, 1]
    min_val = image.min()
    max_val = image.max()
    image = (image - min_val) / (max_val - min_val)

    return image


def get_args():
    # gaussian kernel size for post processing before edge nms
    parser = argparse.ArgumentParser(description='Test output')
    parser.add_argument('--kernel_size', type=int, default=0,
                        help='kernel size')
    args = parser.parse_args()
    return args


def main():
    device = "cuda"
    sam = sam_model_registry["default"](
        checkpoint="/working/model/sam_vit_h_4b8939.pth")
    sam.to(device=device)
    generator = SamAutomaticMaskAndProbabilityGenerator(sam)

    args = get_args()
    kernel_size = args.kernel_size

    # make output directory
    outut_dir = Path(f'output/pred/kernel_size{kernel_size}/test')
    if outut_dir.exists():
        import shutil
        shutil.rmtree(outut_dir)
    outut_dir.mkdir(parents=True)

    img_dir = Path('/working/data/BSR_bsds500/BSR/BSDS500/data/images/test')
    for img_path in tqdm(img_dir.glob('*.jpg')):
        name = img_path.stem
        image = cv2.imread(str(img_path))
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
        if kernel_size > 0:
            assert kernel_size % 2 == 1
            edges = cv2.GaussianBlur(edges, (kernel_size, kernel_size), 0)
        edge_detection = cv2.ximgproc.createStructuredEdgeDetection(
            '/working/model/model.yml.gz')
        orimap = edge_detection.computeOrientation(edges)
        edges = edge_detection.edgesNms(edges, orimap)
        edges = (edges * 255).astype(np.uint8)
        cv2.imwrite(str(outut_dir / f'{name}.png'), edges)


if __name__ == "__main__":
    main()
