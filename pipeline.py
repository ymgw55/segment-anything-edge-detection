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

    parser = argparse.ArgumentParser(description='Test output')

    # dataset
    parser.add_argument('--dataset', type=str, help='BSDS500 or NYUDv2')
    parser.add_argument('--data_split', type=str, default='test',
                        help='train, val, or test')

    # arguments for SAM
    parser.add_argument('--points_per_side', type=int, default=16,
                        help='Number of points per side.')
    parser.add_argument('--points_per_batch', type=int, default=64,
                        help='Number of points per batch')
    parser.add_argument('--pred_iou_thresh', type=float, default=0.00,
                        help='Prediction IOU threshold')
    parser.add_argument('--stability_score_thresh', type=float, default=0.00,
                        help='Stability score threshold')
    parser.add_argument('--stability_score_offset', type=float, default=1.0,
                        help='Stability score offset')
    parser.add_argument('--box_nms_thresh', type=float, default=0.7,
                        help='NMS threshold for box suppression')
    parser.add_argument('--crop_n_layers', type=int, default=0,
                        help='Number of layers to crop')
    parser.add_argument('--crop_nms_thresh', type=float, default=0.7,
                        help='NMS threshold for cropping')
    parser.add_argument('--crop_overlap_ratio', type=float, default=512/1500,
                        help='Overlap ratio for cropping')
    parser.add_argument('--crop_n_points_downscale_factor',
                        type=int, default=1,
                        help='Downscale factor for number of points in crop')
    parser.add_argument('--min_mask_region_area', type=int, default=0,
                        help='Minimum mask region area')
    parser.add_argument('--output_mode', type=str, default="binary_mask",
                        help='Output mode of the mask generator')
    parser.add_argument('--nms_threshold', type=float, default=0.7,
                        help='NMS threshold')
    parser.add_argument('--bzp', type=int, default=0,
                        help='boundary zero padding')

    # gaussian kernel size for post processing before edge nms
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='kernel size')

    args = parser.parse_args()
    return args


def make_output_dir(args):
    dataset = args.dataset
    data_split = args.data_split

    outut_root_dir = Path('output') / dataset
    outut_root_dir.mkdir(parents=True, exist_ok=True)

    last_exp_num = 0
    for exp_dir in outut_root_dir.glob('exp*'):
        exp_num = int(exp_dir.stem[3:])
        last_exp_num = max(last_exp_num, exp_num)
    
    output_dir = \
        outut_root_dir / f'exp{str(last_exp_num + 1).zfill(3)}' / data_split
    output_dir.mkdir(parents=True, exist_ok=True)

    # save args as a text file
    with open(output_dir / 'args.txt', 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k}: {v}\n')

    return output_dir


def main():

    args = get_args()
    dataset = args.dataset
    assert dataset in ["BSDS500", "NYUDv2"]
    data_split = args.data_split
    # assert data_split in ["train", "val", "test"]

    device = "cuda"
    sam = sam_model_registry["default"](
        checkpoint="model/sam_vit_h_4b8939.pth")
    sam.to(device=device)
    generator = SamAutomaticMaskAndProbabilityGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        stability_score_offset=args.stability_score_offset,
        box_nms_thresh=args.box_nms_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_nms_thresh=args.crop_nms_thresh,
        crop_overlap_ratio=args.crop_overlap_ratio,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        min_mask_region_area=args.min_mask_region_area,
        output_mode=args.output_mode,
        nms_threshold=args.nms_threshold
    )

    kernel_size = args.kernel_size

    # make output directory
    output_dir = make_output_dir(args)

    img_dir = Path('data') / dataset / 'images' / data_split

    if dataset == 'BSDS500':
        suf = 'jpg'
    else:
        suf = 'png'

    for img_path in tqdm(list(img_dir.glob(f'*.{suf}'))):
    
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
            'model/model.yml.gz')
        orimap = edge_detection.computeOrientation(edges)
        edges = edge_detection.edgesNms(edges, orimap)
        edges = (edges * 255).astype(np.uint8)

        cv2.imwrite(str(output_dir / f'{name}.png'), edges)


if __name__ == "__main__":
    main()
