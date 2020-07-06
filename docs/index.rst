.. tensorbank documentation master file, created by
   sphinx-quickstart on Sun Jul  5 21:04:06 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TensorBank
==========

TensorBank is a collection of assorted algorithms expressed in Tensors so that
they can be quickly executed on batched data in a variety of ML workflows.

Every function is supported in one or more of the following backends:

* TensorFlow
* PyTorch
* NumPy

.. code-block:: python

    import tensorflow as tf
    import tensorbank.tf as tb

    tb.axis_aligned_boxes.area(
       [[1, 1, 2, 2],
        [-1, -1, 1, 2]])
    >>> tf.Tensor([1 6], shape=(2,), dtype=int32)

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   ./reference/index



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
