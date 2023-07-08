# Python port of BSDS 500 boundary prediction evaluation suite

Uses quite a lot of code from the original BSDS evaluation suite at
[berkeley.edu/Research/Projects/CS/vision/bsds/](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)


Takes the original C++ source code that provides the `matchPixels` function for Matlab
and wraps it with Cython to make it available from Python.

Provides a Python implementation of the morphological thinning operation.

Compile the extension module with:

`python setup.py build_ext --inplace`

Then run:

`python verify.py <path_to_bsds500_root_directory>`

You should get output that (almost) matches the text files in the
`bench/data/test_2` directory within the BSDS package.
