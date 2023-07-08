import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from bsds.bsds_dataset import BSDSDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Test output')
    parser.add_argument('bsds_path', type=str,
                        help='the root path of the BSDS-500 dataset')
    parser.add_argument('--val_test', type=str, default='test',
                        help='val or test')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    bsds_path = args.bsds_path
    val_test = args.val_test

    ds = BSDSDataset(bsds_path)

    if val_test == 'val':
        SAMPLE_NAMES = ds.val_sample_names
    elif val_test == 'test':
        SAMPLE_NAMES = ds.test_sample_names
    else:
        print('need to specify either val or test, not {}'.format(val_test))
        sys.exit()

    def progress(x, *args):
        return x

    output_dir = f'../output/gt'
    os.makedirs(output_dir, exist_ok=True)
    for name in tqdm(progress(SAMPLE_NAMES)):
        # Get the paths for the ground truth and predicted boundaries

        # Load them
        gt_b = ds.boundaries(name)
        gt_b = np.mean(gt_b, axis=0)
        plt.imsave(os.path.join(output_dir, f'{name}.png'), gt_b, cmap='binary')


if __name__ == '__main__':
    main()
