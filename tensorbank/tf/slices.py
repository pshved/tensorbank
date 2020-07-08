"""Advanced Tensor slicing
==========================

Utilities for advanced tensor slicing and batching operations.


Reference
---------
"""
import tensorflow as tf

def slice_within_stride(x, stride, si=0, ei=None, keepdims=True):
    """Select ``x[..., (i * stride + si):(i * stride + ei)]`` for each i.

    The tensor returned will have the last dimension shrunk by a factor of
    ``(ei-si)/stride``.

    As a natural special case, ``tb.multiple_within_stride(x, N)`` is
    equivalent to adding a dimension of ``N`` at the end, as in
    ``tf.expand_dims(x, (..., -1, N))``.

    Example:

        When predicting anchor positions in SSD, ``num_classes +
        num_offsets`` are predicted for each anchor.  To get only the
        class confidence, this would be used::

            logits = model(input)
            class_logits = tb.slice_within_stride(
                logits,
                0,
                num_classes,
                num_classes + num_offsets)
            loss = softmax_cross_entropy_with_logits(
                class_preds, class_logits)

    Args:
        x (tf.Tensor): value to modify
        stride (int): stride for the last dimension
        si (int): starting index within stride.  Negative indices are
            supported.  Defaults to 0.
        ei (int): end index (1 element after the last) within stride.
           Negative indices are supported. Defaults to ``None``, which
           means "until the last element".
        keepdims (bool): if False, adds another dimension that
            iterates over each stride.  This dimension will be of size
            ``ei-si``.  Defaults to True.

    Returns:
        tf.Tensor: modified ``x`` with the last dimension sliced.

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

