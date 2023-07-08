# segment-anything-edge-detection

<p>
<img src='assets/fish.jpg' height=150px /> <img src='assets/fish_gt.png' height=150px /> <img src='assets/edge.png' height=150px />
</p>

This repository provides code for performing edge detection using the Automatic Mask Generation (AMG) of the Segment Anything Model (SAM). Since the code used in the paper is not currently available to the public, this implementation is based on the descriptions provided in the paper.

The image on the left is taken from the BSDS. The middle is the ground truth edge. The image on the right is the result of applying edge detection.

## Requirements
This repository assumes that you can already use a SAM model. 
Note that this repository uses `opencv-contrib-python`, not `opencv-python`, so install it as follows:
```bash
pip install opencv-contrib-python
```
See [the description](https://pypi.org/project/opencv-contrib-python/) for more details.

You will also need to download the model for Edge NMS beforehand.
```bash
cd /working/model
wget https://cdn.rawgit.com/opencv/opencv_extra/3.3.0/testdata/cv/ximgproc/model.yml.gz
```

## Run
To generate the image above, do the following:
```
python example.py
```
The output result is generated in `output/pred/example`.

Assuming the BSDS path is `/working/data/BSR_bsds500/BSR/`, to generate output for the bsds500 test-set, do the following:
```
python test.py
```
The output result is generated in `output/pred/${pp_name}/test`.

# Evaluation
We use [py-bsds500](https://github.com/Britefury/py-bsds500/tree/master) for edge detection. Some bugs have been fixed and ported to the `py-bsds500` directory.
Compile the extension module with:
```bash
cd py-bsds500
python setup.py build_ext --inplace
```
Then run:
```bash
# This will take about 3.5 hours.
python evaluate.py /working/data/BSR_bsds500/BSR/ ../output/pred/${pp_name} test --thresholds 99
```

# Todo
- Faster calculation of the evaluation through parallel processing.

## Reference
The code in this repository mainly uses code from the following two repositories. Thank you.
- [segment-anything](https://github.com/facebookresearch/segment-anything)
- [py-bsds500](https://github.com/Britefury/py-bsds500/tree/master)