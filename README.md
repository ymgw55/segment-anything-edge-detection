# segment-anything-edge-detection

<p>
<img src='assets/fish.jpg' height=150px /> <img src='assets/edge.png' height=150px />
</p>

This repository provides code for performing edge detection using the Automatic Mask Generation (AMG) of the Segment Anything Model (SAM). Since the code used in the paper is not currently available to the public, this implementation is based on the descriptions provided in the paper.

The image on the far left is taken from the BSDS dataset. <!-- The center is the ground truth edge.  --> The final image on the right is the result of applying edge detection using probability map thresholds.

## Run
This repository assumes that you can already use a SAM model.
To generate the image above, do the following:
```
python main.py --pp threshold
```

## TODO
- The thick lines on the probability map split into two lines after Edge NMS, so there is a need to improve the accuracy. Therefore, edge detection using thresholds for probability maps is currently the default setting. If you know something about edge detection, feel free to submit a pull request.
- If time permits, I may create a pipeline to perform an end-to-end evaluation of SAM's edge detection performance using the BSDS dataset.

## Reference
The code in this repository mainly uses code from the following two repositories. Thanks you.
- [segment-anything](https://github.com/facebookresearch/segment-anything)