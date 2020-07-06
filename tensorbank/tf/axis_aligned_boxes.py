"""Axis-aligned Bounding Boxes
==============================

Axis-aligned rectangular bounding boxes have their sides, natuarally, aligned
with coordinate axes in the multidimensional Eucledian space R^D.  Axis-aligned
boxes have fewer parameters than arbitrarily rotated boxes, which simplifies
their learning and operations.

A D-dimensional axis-aligned box in R^D is represented as a 2*D array of the
coordinates of the bottom-left corner followed by the coordinates of the
top-right corner. E.g. a two-dimensional box would have the coordinates laid
out like so:

    y1, x1, y2, x2

Box format
----------

Throughout this module, all boxes are defined in "matrix convention" also known
as "ij-indexed".

In some other APIs (e.g. in Matplotlib), two-dimensional boxes are defined in
the "cartesian" notation aka "xy".  Be very careful and transpose these boxes
when using this API.

In a correct box, each component of the bottom corner will be smaller or equal
than the corresponding component of top corner.  The boxes where this is not
the case are degenerate.  It is undefined what values are returned for the
degenerate boxes, but no exception will be raised.

API
----------
"""
import tensorflow as tf

def intersection_area(a, b):
    """Computes intersection area of each pair of boxes in a and b.

    This function is primarily intended to use with batched anchor matching.
    If the number of boxes in each batch is different, simply pad the boxes
    with 0.0 and ignore the rows.

    Args:
        a (Tensor [N x K x 2*D]): box coordinates.  N is batch size, K is the
            number of boxes in a batch, D is the dimension of the euclidian
            space.  

        b (Tensor [N x M x 2*D]): box coordinates, N is batch size, M is the
            number of boxes in a batch, D is the dimension of the euclidian
            space.  See above for more details.

    Returns:
        Tensor [N x K x M] of pairwise box intersection areas using the
        standard volume metric in R^D.

    Example:
        tf.axis_aligned
    """

    a = tf.convert_to_tensor(a)
    b = tf.convert_to_tensor(b)

    # Shape check
    assert len(b.shape) == 3, "Wrong shape of b: got {} expect 3 components".format(b.shape)
    assert a.shape[0] == b.shape[0]
    assert a.shape[2] == b.shape[2]
    N, K, Dx2 = a.shape
    N, M, _ = b.shape
    D = Dx2 // 2
    assert 2*D == Dx2

    # Extend a-s so we can broadcast transposed b-s
    a_ext = a                                      # N x K x 2D
    a_ext = tf.expand_dims(a_ext, 2)               # N x K x 1 x 2D

    # Using tf.broadcast_to instead of tf.tile for extra speed.
    # (To be honest, I never benchmarked.)
    a_ext = tf.broadcast_to(a_ext, (N, K, M, 2*D)) # N x K x M x 2D

    a1 = a_ext[:, :, :, :D]                        # N x K x M x D
    a2 = a_ext[:, :, :, D:]                        # N x K x M x D
    b1 = tf.expand_dims(b[:, :, :D], 1)            # N x 1 x M x D
    b2 = tf.expand_dims(b[:, :, D:], 1)            # N x 1 x M x D 

    i1 = tf.maximum(a1, b1)                        # N x K x M x D
    i2 = tf.minimum(a2, b2)                        # N x K x M x D
    ds = tf.maximum(i2 - i1, tf.zeros_like(i1))    # N x K x M x D

    # Got the intersection box lengths, now compute the area.
    return tf.math.reduce_prod(ds, axis=3)         # N x K x M


def area(a):
    """Computes area of each box in a.

    This function is primarily intended to use with batched box matching.

    Args:
        a (Tensor [...dims... x 2*D]): box coordinates where  D is the dimension of
        the euclidian space.  Each box is represented as the coordinates of the
        bottom left corner followed by the coordinates of the top right corner.
        E.g. a two-dimensional box would have the coordinates laid out like so:

            x1, y1, x2, y2

    Returns:
        Tensor [...dims...]: box volumes using the standard volume metric in R^D.
    """
    # Shape check
    shapes = list(a.shape)
    D = shapes[-1] // 2
    assert shapes[-1] == 2*D

    # Compuite the slice while keeping dims.  Essentially we want the :D and D:
    # on the last dimension.
    bottom_begin = [0 for _ in shapes]
    bottom_size = shapes
    bottom_size[-1] = D
    top_begin = [0 for _ in shapes]
    top_begin[-1] = D
    top_size = shapes
    top_size[-1] = D

    bottom = tf.slice(a, bottom_begin, bottom_size)
    top = tf.slice(a, top_begin, top_size)

    return tf.reduce_prod(top - bottom, axis=-1)

def iou(a, b):
    """Computes intersection over union of each pair of boxes in a and b.

    This function is primarily intended to use with batched anchor matching.
    If the number of boxes in each batch is different, simply pad the boxes
    with 0.0 and ignore the rows.

    When the boxes do not intersect, their IOU is 0.0.  When a pair of boxes
    has the union area of 0.0 (e.g. when both boxes are empty) their IoU will
    be NaN.

    Args:
        a: Tensor [N x K x 2*D] of box coordinates.  N is batch size, K is the
        number of boxes in a batch, D is the dimension of the euclidian space.
        Each box is represented as the coordinates of the bottom left corner
        followed by the coordinates of the top right corner.  E.g. a
        two-dimensional box would have the coordinates laid out like so:

            x1, y1, x2, y2

        b: Tensor [N x M x 2D] of box coordinates, N is batch size, M is the
        number of boxes in a batch, D is the dimension of the euclidian space.
        See above for more details.

    Returns:
        Tensor [N x K x M] of pairwise box IoUs using the standard volume
        metric in R^D.
    """

    # Shape check
    assert len(b.shape) == 3, "Wrong shape of b: {}".format(b.shape)
    assert a.shape[0] == b.shape[0]
    assert a.shape[2] == b.shape[2]
    N, K, Dx2 = a.shape
    N, M, _ = b.shape
    D = Dx2 // 2
    assert 2*D == Dx2

    i = intersection_area(a, b)   # N x K x M
    area_a = area(a)   # N x K
    area_b = area(b)   # N x M

    # Compute pairwise union.  Repeat each tensor along the orthogonal dimension.
    area_a = tf.broadcast_to(tf.expand_dims(area_a, 2), (N, K, M))   # N x K x M
    area_b = tf.broadcast_to(tf.expand_dims(area_b, 1), (N, K, M))   # N x K x M

    u = area_a + area_b - i

    return i / u


def evenly_spaced(box_counts, box_sizes, image_shape, offset=None, dtype=tf.float32):
    """Returns "anchor" boxes evenly spaced within the image.

    We assume that the image is D-dimensional, and give examples for 2
    dimensions.  The length of all lists is equivalent to the number of scales S
    in the detector.  Boxes for each scale are appended after the previous scale.

    Please note that the box_sizes are defined in the tensor order.  This is
    different from the common way to define the box sizes in the W,H order for
    2D boxes. 

    Example:

        The following function will return 12 boxes: 3 boxes centered in each
        point of a 2x2 grid:

            tb.axis_aligned_boxes.evenly_spaced(
                [ (2,2) ],
                [ [(10, 15), (15, 10)] ],
                (224, 224),
            )


    Args:
        box_counts (List (length S) of D-tuples): number of elements
            in a grid along each axis for the box centers.  The first image is
            at the offset defined by offset.  Can be a tf.Tensor.
        box_sizes (List (length S) of lists (length B_i) of lists (length D)):
            box sizes for each scale.  This shoudn't be a Tensor since
            different scales can have different number of anchors.
        image_shape: D-tuple that defines the overall image shape.  Can be a
            tf.Tensor.
        offset: List (length S) of D-tuples that define the offset of the first
            image from 0^D.  Can be a tf.Tensor.

    Returns:
        Tensor [number_of_boxes x 2*D]: list of boxes
    
    """
    final_dtype = dtype
    # Compute all boxes with large precision regardless of the final representation.
    intermediate_dtype = tf.float32

    # Shape check
    # S x 2D
    S, D = (len(box_counts), len(box_counts[0]) if len(box_counts) else 0)
    assert len(box_sizes) == S, \
            "len(box_sizes] is {}, but needs to be equal to box_counts.shape[0] which is {}".format(
                    len(box_sizes), S)
    assert len(image_shape) == D, \
            "len(image_shape) is {} (image_shape is {}), but needs to be equal to box_counts.shape[1] which is {}".format(
                    len(image_shape), image_shape, D)

    boxes_for_all_scales = []

    for s, box_sizes_for_shape in enumerate(box_sizes):
        #assert len(box_sizes_for_shape[2]) == D, \
                #"len(box_sizes[{}]) is {}, but needs to be equal to 2*box_counts.shape[1] which is {}".format(
                        #s, len(box_sizes_for_shape), D)
        # Prepare range for all dimensions
        grid_elements = []
        for d, size_along_d in enumerate(box_counts[s]):
            delta = image_shape[d] / size_along_d
            # Note: the first argument is the max value rather than the count.
            grid_elements.append(tf.range(tf.cast(image_shape[d], dtype=intermediate_dtype), delta=delta))

        # This one little trick prepares a d-dimensional grid with evenly spaced things.
        # We use indexing = 'ij' (the matrix convention), so that it will first
        # iterate over the 1st row, then the 2nd, etc.  The reason for this
        # indexing convention is that tf.keras.Flatten after tf.keras.Conv2D
        # iterates in the same manner.
        # Shape: box_counts[s]... x D
        ixs = tf.cast(tf.stack(tf.meshgrid(*grid_elements, indexing='ij'), axis=-1),
                dtype=intermediate_dtype)

        # TODO: add offset!
        assert offset is None

        # Shape: B_s x D
        ixs = tf.reshape(ixs, (-1, D))
        all_ixs = []
        for box_size in box_sizes_for_shape:
            # Shape: D
            bs = tf.convert_to_tensor(box_size, dtype=intermediate_dtype)
            # Add bottom point and top point assuming ixs is the center.
            all_ixs += [ixs - bs / 2, ixs + bs / 2]

        # Now all_ixs are the indices of all boxes for the grid.  Stack them along the last axis.
        # Shape of the new element: B_s x (box_sizes * 2D)
        all_boxes = tf.concat(all_ixs, axis=-1)
        # But!  Now we reshape to : (B_s * box_sizes) x 2D
        # This is not the same as tf.concat(all_ixs, axis=0)!
        boxes_for_all_scales.append(tf.reshape(all_boxes, (-1, 2*D)))

    # Preapare coordinates of all shapes
    return tf.cast(tf.concat(boxes_for_all_scales, axis=0), dtype=final_dtype)
