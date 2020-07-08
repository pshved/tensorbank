"""Advanced Tensor slicing
==========================


Reference
---------
"""
import tensorflow as tf

def slice_within_stride(x, stride, si=0, ei=None, keepdims=True):
    """Select x[..., (i * stride + si):(i * stride + ei)] for each i.

    The tensor returned will have the last dimension shrunk by a factor of
    (ei-si)/stride.

    As a natural special case, ``tb.multiple_within_stride(x, 10)`` will simply
    reshape the tensor by adding a dimension of size 10 at the end of ``x``'s
    shape. 

    Args:
        x (Tensor): value to modify
        stride (int): stride for the last timension
        si (int): starting index within stride.  Defaults to 0.
        ei (int): end indes (1 element after the last) within stride.  Defaults
            to ``None``, which means "until the last element".
        keepdims (bool, default: True): if False, adds another dimension that
            iterates over each stride.

    """
    step1 = tf.reshape(x, (-1, stride))
    step2 = step1[..., si:ei]
    new_shape = list(x.shape)
    new_shape[-1] = -1

    if not keepdims:
        if ei is None:
            ei = stride
        # Calculate the size of the slice.  This is O(stride) which is
        # small.
        last_dim_len = len(list(range(stride)[si:ei]))
        new_shape.append(last_dim_len)

        print("NS: {}".format(new_shape))

    step3 = tf.reshape(step2, new_shape)
    return step3

