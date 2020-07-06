# TensorBank

[![Build Status](https://travis-ci.com/pshved/tensorbank.svg?branch=master)](https://travis-ci.com/pshved/tensorbank)

TensorBank is a collection of assorted algorithms expressed in Tensors.

We do not intend to limit ourselves to a specific domain.  The initial batch of
algorithms is focused on point and box gemoetry for some object detection
tasks, but more algorithms will be added later.

We are open to all backends, including Tensorflow, Pytorch, and NumPy.

Primarily, this project is to avoid copy-pasting the "utils" directory from one
project to the next :-)

## Installation

```
pip install tensorbank
```

## Usage

If you're using TensorFlow, import TensorBank as follows and use the `tb.`
prefix:

```python
import tensorbank.tf as tb

tb.axis_aligned_boxes.area(
	 [[1, 1, 2, 2],
		[-1, -1, 1, 2]])
>>> tf.Tensor([1 6], shape=(2,), dtype=int32)
```

See API Reference on Readthedocs for more exampels and the comprehensive usage.

