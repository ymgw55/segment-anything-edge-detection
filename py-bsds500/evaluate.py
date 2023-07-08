import argparse
import os
import pickle
import sys

import numpy as np
import tqdm
from skimage.io import imread
from skimage.util import img_as_float

from bsds import evaluate_boundaries
from bsds.bsds_dataset import BSDSDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Test output')
    parser.add_argument('bsds_path', type=str,
                        help='the root path of the BSDS-500 dataset')
    parser.add_argument('pred_path', type=str,
                        help='the root path of the predictions')
    parser.add_argument('val_test', type=str,
                        help='val or test')
    parser.add_argument('--thresholds', type=str, default='99',
                        help='the number of thresholds')
    parser.add_argument('--suffix_ext', type=str, default='.png',
                        help='suffix and extension')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    bsds_path = args.bsds_path
    pred_path = args.pred_path
    val_test = args.val_test
    suffix_ext = args.suffix_ext
    thresholds = args.thresholds
    thresholds = thresholds.strip()
    try:
        n_thresholds = int(thresholds)
        thresholds = n_thresholds
    except ValueError:
        try:
            if thresholds.startswith('[') and thresholds.endswith(']'):
                thresholds = thresholds[1:-1]
                thresholds = np.array(
                    [float(t.strip()) for t in thresholds.split(',')])
            else:
                print('Bad threshold format; '
                      'should be a python list of floats (`[a, b, c]`)')
                sys.exit()
        except ValueError:
            print('Bad threshold format; '
                  'should be a python list of ints (`[a, b, c]`)')
            sys.exit()

    ds = BSDSDataset(bsds_path)

    if val_test == 'val':
        SAMPLE_NAMES = ds.val_sample_names
    elif val_test == 'test':
        SAMPLE_NAMES = ds.test_sample_names
    else:
        print('need to specify either val or test, not {}'.format(val_test))
        sys.exit()

    def load_gt_boundaries(sample_name):
        return ds.boundaries(sample_name)

    def load_pred(sample_name):
        sample_path = os.path.join(pred_path, f'{sample_name}{suffix_ext}')
        pred = img_as_float(imread(sample_path))
        bnds = ds.boundaries(sample_name)
        tgt_shape = bnds[0].shape
        pred = pred[:tgt_shape[0], :tgt_shape[1]]
        pred = np.pad(pred, [(0, tgt_shape[0]-pred.shape[0]),
                             (0, tgt_shape[1]-pred.shape[1])], mode='constant')
        return pred

    stem = os.path.basename(pred_path)
    output_dir = f'../output/result/{stem}/{val_test}'
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(f'{output_dir}',
                                f'results_thr{thresholds}.pkl')
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        SAMPLE_NAMES, sample_results, threshold_results, overall_result = \
            results
    else:
        sample_results, threshold_results, overall_result = \
            evaluate_boundaries.pr_evaluation(
                thresholds, SAMPLE_NAMES, load_gt_boundaries,
                load_pred, progress=tqdm.tqdm)
        results = (SAMPLE_NAMES, sample_results,
                   threshold_results, overall_result)
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)

    ods = overall_result.f1
    ois = overall_result.best_f1

    rs = []
    ps = []
    for res in threshold_results:
        rs.append(res.recall)
        ps.append(res.precision)
    ap = np.trapz(ps[::-1], rs[::-1])

    threshold_index = next(i for i, p in enumerate(ps) if p >= 0.5)
    r50 = rs[threshold_index]

    with open(os.path.join(f'{output_dir}', f'results_thr{thresholds}.txt'),
              'w') as f:
        text = f'ODS: {ods:.3f}, OIS: {ois:.3f} AP: {ap:.3f} R50: {r50:.3f}'
        print(text, file=f)
        print(text)


if __name__ == '__main__':
    main()
