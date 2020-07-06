"""Point operations
===================

Batch operations on points in D-dimensional Eucledian space R^D.

"""

import tensorflow as tf

def pairwise_l2_distance(a, b, sqrt=True):
    """Compute pairwise L2 distance between all points in A and B.

    L2 norm is ``sqrt(sum(|x_i - y_i| ^ 2))``

    Args:
        a (Tensor [N x K x D]): point coordinates.  N is batch size, K is the
            number of points in a batch, D is the dimension of the Euclidian
            space.

        b (Tensor [N x M x D]): point coordinates, N is batch size, M is the
            number of points in a batch, D is the dimension of the euclidian
            space.

        sqrt (bool, optional): whether take the square root. Defaults to True.

    Returns:
        Tensor [N x K x M]: pairwise L2 distance between each pair of points.
    
    """

    a = tf.convert_to_tensor(a)
    b = tf.convert_to_tensor(b)

    # (a_i - b_j)^2 = a_i^2 + b_j^2 - 2 * a_i * b_j
    a_ext = tf.expand_dims(a, 2)                   # N x K x 1 x D
    b_ext = tf.expand_dims(b, 1)                   # N x 1 x M x D
    a2 = tf.reduce_sum(a_ext * a_ext, axis=3)      # N x K x 1
    b2 = tf.reduce_sum(b_ext * b_ext, axis=3)      # N x 1 x M
    ab = tf.matmul(a, tf.transpose(b, (0, 2, 1)))  # N x K x M
    L2_square = a2 + b2 - 2 * ab                   # N x K x M
    if sqrt:
        return L2_square ** 0.5
    else:
        return L2_square


def pairwise_l1_distance(a, b):
    """Compute pairwise L1 distance between all points in A and B.

    L1 norm is ``sum(|x_i - y_i|)``

    Args:
        a (Tensor [N x K x D]): point coordinates.  N is batch size, K is the
            number of points in a batch, D is the dimension of the Euclidian
            space.

        b (Tensor [N x M x D]): point coordinates, N is batch size, M is the
            number of points in a batch, D is the dimension of the euclidian
            space.

    Returns:
        Tensor [N x K x M]: pairwise L1 distance between each pair of points.
    
    """

    a = tf.convert_to_tensor(a)
    b = tf.convert_to_tensor(b)

    a_ext = tf.expand_dims(a, 2)  # N x K x 1 x D
    b_ext = tf.expand_dims(b, 1)  # N x 1 x M x D
    return tf.reduce_sum(tf.abs(a_ext - b_ext), axis=3)   # N x K x M


def pairwise_l_inf_distance(a, b):
    """Compute pairwise L∞ distance between all points in A and B.

    L-infinity is ``max(|x_i - y_i|)``

    Args:
        a (Tensor [N x K x D]): point coordinates.  N is batch size, K is the
            number of points in a batch, D is the dimension of the Euclidian
            space.

        b (Tensor [N x M x D]): point coordinates, N is batch size, M is the
            number of points in a batch, D is the dimension of the euclidian
            space.

    Returns:
        Tensor [N x K x M]: pairwise L-infinity distance between each pair of
        points.
    
    """

    a = tf.convert_to_tensor(a)
    b = tf.convert_to_tensor(b)

    a_ext = tf.expand_dims(a, 2)  # N x K x 1 x D
    b_ext = tf.expand_dims(b, 1)  # N x 1 x M x D
    return tf.reduce_max(tf.abs(a_ext - b_ext), axis=3)   # N x K x M
