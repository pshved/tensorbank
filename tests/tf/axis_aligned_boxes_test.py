import sys
import numpy as np
import unittest

import tensorbank.tf as tb

# Let's reimplement the algorithms without tensors or batches.
def nontf_intersection(a, b):
    """a and b are numpy arrays.  Produces intersection of the two boxes."""
    D = len(a) // 2
    assert D > 0
    assert len(b) == 2*D
    a1 = a[:D]
    a2 = a[D:]
    b1 = b[:D]
    b2 = b[D:]

    i1 = np.maximum(a1, b1)
    i2 = np.minimum(a2, b2)
    ds = np.maximum(i2 - i1, np.zeros_like(i1))

    return np.product(ds)
    
def nontf_area(a):
    D = len(a) // 2
    assert D > 0
    assert len(a) == 2*D
    a1 = a[:D]
    a2 = a[D:]

    ds = np.maximum(a2 - a1, np.zeros_like(a1))

    return np.product(ds)

def nontf_iou(a, b):
    ix = nontf_intersection(a, b)
    union = nontf_area(a) + nontf_area(b) - ix
    return ix / union
    

def make_random_boxes(shape_1, shape_2):
    """Makes random boxes such that many of them intersect."""

    start_1 = np.random.rand(*shape_1) * 0.5
    size_1 = np.random.rand(*shape_1) * 0.5 
    boxes_1 = np.concatenate( [start_1, start_1 + size_1], axis=2)

    start_2 = np.random.rand(*shape_2) * 0.5 + 0.3
    size_2 = np.random.rand(*shape_2) * 0.5 
    boxes_2 = np.concatenate( [start_2, start_2 + size_2], axis=2)

    return boxes_1, boxes_2


class NontfIntersectionTest(unittest.TestCase):
    def testBoxIntersectionArea(self):
        box1 = [1.0, 1.0, 3.0, 3.0]
        box2 = np.array([2.0, 2.0, 4.0, 4.0])
        self.assertAlmostEqual(4.0, nontf_intersection(box1, box1))
        self.assertAlmostEqual(1.0, nontf_intersection(box1, box2))
        self.assertAlmostEqual(1.0, nontf_intersection(box2, box1))

    def testBox1D(self):
        box1 = np.array([1.0, 2.0])
        box2 = np.array([1.5, 2.5])
        self.assertAlmostEqual(0.5, nontf_intersection(box1, box2))
        self.assertAlmostEqual(0.5, nontf_intersection(box2, box1))

    def testZeroIx(self):
        box1 = np.array([1.0, 2.0])
        box2 = np.array([3.0, 4.0])
        self.assertAlmostEqual(0.0, nontf_intersection(box1, box2))

    def testZeroIxTouch(self):
        box1 = np.array([1.0, 2.0])
        box2 = np.array([2.0, 3.0])
        self.assertAlmostEqual(0.0, nontf_intersection(box1, box2))

class NontfAreaTest(unittest.TestCase):
    def testBoxArea(self):
        box1 = np.array([1.0, 1.0, 3.0, 3.0])
        box2 = np.array([2.0, 2.0, 4.0, 4.0])
        self.assertAlmostEqual(4.0, nontf_area(box1))
        self.assertAlmostEqual(4.0, nontf_area(box2))

class NontfIoUTest(unittest.TestCase):
    def testBoxes(self):
        box1 = np.array([1.0, 1.0, 3.0, 3.0])
        box2 = np.array([2.0, 2.0, 4.0, 4.0])
        self.assertAlmostEqual(1.0, nontf_iou(box1, box1))
        self.assertAlmostEqual(1.0 / 7.0, nontf_iou(box1, box2))

class IntersectionAreaTest(unittest.TestCase):
    def compute_good_answer(self, boxes_1, boxes_2):
        r = []
        for batch_i, boxes_1_batch in enumerate(boxes_1):
            boxes_2_batch = boxes_2[batch_i]
            for box_1 in boxes_1_batch:
                for box_2 in boxes_2_batch:
                    r.append(nontf_intersection(box_1, box_2))

        return np.reshape(r, (boxes_1.shape[0], boxes_1.shape[1], boxes_2.shape[1]))

    def testIntersectionSimple(self):
        # Test 1 batch of 2 boxes vs 3 boxes
        # Not numpy to test the auto conversion.
        boxes1 = ([
            [[1.0, 1.5, 3.0, 3.0],
             [2.0, 2.5, 4.0, 4.0],
            ]
        ])
        boxes2 = [
            [[2.0, 2.5, 4.0, 4.0],
             [1.0, 1.5, 3.0, 3.0],
             [0.0, 0.0, 0.0, 0.0],
            ],
        ]

        want = np.array([
            [[0.5, 3.0, 0.0],
             [3.0, 0.5, 0.0],
            ],
        ])

        self.assertEqual(np.array(boxes1).shape, (1, 2, 4))
        self.assertEqual(np.array(boxes2).shape, (1, 3, 4))
        self.assertEqual(want.shape, (1, 2, 3))

        got = tb.axis_aligned_boxes.intersection_area(boxes1, boxes2)

        np.testing.assert_array_almost_equal(got, want)

        # Test our box utils too.
        got_via_box_utils = self.compute_good_answer(np.array(boxes1), np.array(boxes2))
        np.testing.assert_almost_equal(got_via_box_utils, want)


    def testIntersectionRandom1d(self):
        boxes_1, boxes_2 = make_random_boxes( (10, 2, 1), (10, 5, 1) )
        assert boxes_1.shape == (10, 2, 2)
        want = self.compute_good_answer(boxes_1, boxes_2)
        got = tb.axis_aligned_boxes.intersection_area(boxes_1, boxes_2)
        np.testing.assert_almost_equal(got, want)

    def testIntersectionRandom2d(self):
        BATCH_SIZE = 10
        boxes_1, boxes_2 = make_random_boxes( (BATCH_SIZE, 2000, 2), (BATCH_SIZE, 10, 2) )
        assert boxes_1.shape == (BATCH_SIZE, 2000, 4)
        print("Computing the right answer using for-loops (will take 10 sec)...")
        want = self.compute_good_answer(boxes_1, boxes_2)
        print("Computing the right answer using Tensors...")
        got = tb.axis_aligned_boxes.intersection_area(boxes_1, boxes_2)

        found_nonzero = np.count_nonzero(got)
        print("done, found {} nonzero boxes!".format(found_nonzero))
        assert found_nonzero > 10000
        np.testing.assert_almost_equal(got, want)

    def testIntersectionRandom3d(self):
        boxes_1, boxes_2 = make_random_boxes( (17, 5, 3), (17, 11, 3) )
        assert boxes_1.shape == (17, 5, 6)
        want = self.compute_good_answer(boxes_1, boxes_2)
        got = tb.axis_aligned_boxes.intersection_area(boxes_1, boxes_2)
        np.testing.assert_almost_equal(got, want)

    def testIntersectionEmpty(self):
        # Test 1 batch of 2 boxes vs 3 boxes
        # Not numpy to test the auto conversion.
        boxes1 = [
            [[1.0, 1.5, 3.0, 3.0],
             [2.0, 2.5, 4.0, 4.0],
            ]
        ]
        boxes2 = [
            [[5.0, 6.0, 7.0, 8.0],
             [0.0, 0.0, 1.0, 3.0],
            ],
        ]

        want = np.array([
            [[0.0, 0.0],
             [0.0, 0.0],
            ],
        ])

        got = tb.axis_aligned_boxes.intersection_area(boxes1, boxes2)

        np.testing.assert_array_almost_equal(got, want)

        # Test our box utils too.
        got_via_box_utils = self.compute_good_answer(np.array(boxes1), np.array(boxes2))
        np.testing.assert_almost_equal(got_via_box_utils, want)


class IntersectionTest(unittest.TestCase):
    def testIntersectionSimple(self):
        # Test 1 batch of 2 boxes vs 3 boxes
        # Not numpy to test the auto conversion.
        boxes1 = [
            [[1.0, 1.5, 3.0, 3.0],
             [2.0, 2.5, 4.0, 4.0],
            ]
        ]
        boxes2 = [
            [[2.0, 2.5, 4.0, 4.0],
             [1.0, 1.5, 3.0, 3.0],
             [0.0, 0.0, 0.0, 0.0],
            ],
        ]

        want = np.array([
            [[ [2.0, 2.5, 3.0, 3.0], [1.0, 1.5, 3.0, 3.0], [1.0, 1.5, 0.0, 0.0]],
             [ [2.0, 2.5, 4.0, 4.0], [2.0, 2.5, 3.0, 3.0], [2.0, 2.5, 0.0, 0.0]],
            ],
        ])

        self.assertEqual(np.array(boxes1).shape, (1, 2, 4))
        self.assertEqual(np.array(boxes2).shape, (1, 3, 4))
        self.assertEqual(want.shape, (1, 2, 3, 4))

        got = tb.axis_aligned_boxes.intersection(boxes1, boxes2)

        np.testing.assert_array_almost_equal(got, want)

    def testIntersectionOrthogonal(self):
        # Test two boxes where the intersection doesn't have any common corners.
        boxes1 = np.array([
            [[2.0, 2.0, 3.0, 5.0],
            ]
        ])
        boxes2 = np.array([
            [[1.0, 3.0, 4.0, 4.0],
            ],
        ])

        want = np.array([
            [[ [2.0, 3.0, 3.0, 4.0] ]],
        ])

        self.assertEqual(boxes1.shape, (1, 1, 4))
        self.assertEqual(boxes2.shape, (1, 1, 4))
        self.assertEqual(want.shape, (1, 1, 1, 4))

        got = tb.axis_aligned_boxes.intersection(boxes1, boxes2)

        np.testing.assert_array_almost_equal(got, want)


class AreaTest(unittest.TestCase):
    def compute_good_answer(self, boxes_1):
        r = []
        for batch_i, boxes_1_batch in enumerate(boxes_1):
            for box_1 in boxes_1_batch:
                r.append(nontf_area(box_1))

        return np.reshape(r, (boxes_1.shape[0], boxes_1.shape[1]))

    def testSimple(self):
        # Test 1 batch of 2 boxes vs 3 boxes
        # Not numpy to test the auto conversion.
        boxes1 = [
            [[1.0, 1.5, 3.0, 3.0],
             [2.0, 2.5, 4.0, 4.0],
            ]
        ]
        boxes2 = [
            [[2.0, 2.5, 4.0, 4.0],
             [0.0, 0.5, 3.0, 3.0],
             [0.0, 0.0, 0.0, 0.0],
            ],
        ]

        want1 = np.array([
            [3.0, 3.0],
        ])

        want2 = np.array([
            [3.0, 7.5, 0.0],
        ])

        self.assertEqual(np.array(boxes1).shape, (1, 2, 4))
        self.assertEqual(np.array(boxes2).shape, (1, 3, 4))
        self.assertEqual(want1.shape, (1, 2))
        self.assertEqual(want2.shape, (1, 3))

        got1 = tb.axis_aligned_boxes.area(boxes1)
        got2 = tb.axis_aligned_boxes.area(boxes2)
        np.testing.assert_array_almost_equal(got1, want1)
        np.testing.assert_array_almost_equal(got2, want2)

        # Test our box utils too.
        got_via_box_utils = self.compute_good_answer(np.array(boxes1))
        np.testing.assert_almost_equal(got_via_box_utils, want1)


class IoUTest(unittest.TestCase):
    def compute_good_answer(self, boxes_1, boxes_2):
        r = []
        for batch_i, boxes_1_batch in enumerate(boxes_1):
            boxes_2_batch = boxes_2[batch_i]
            for box_1 in boxes_1_batch:
                for box_2 in boxes_2_batch:
                    r.append(nontf_iou(box_1, box_2))

        return np.reshape(r, (boxes_1.shape[0], boxes_1.shape[1], boxes_2.shape[1]))

    def testSimple(self):
        # Test 1 batch of 2 boxes vs 3 boxes
        # Not numpy to test the auto conversion.
        boxes1 = [
            [[1.0, 1.5, 3.0, 3.0],
             [2.0, 2.5, 4.0, 4.0],
            ]
        ]
        boxes2 = [
            [[2.0, 2.5, 4.0, 4.0],
             [1.0, 1.5, 3.0, 3.0],
             [0.0, 0.0, 0.0, 0.0],
            ],
        ]

        want = np.array([
            [[0.5 / 5.5, 1.0,       0.0],
             [1.0,       0.5 / 5.5, 0.0],
            ],
        ])

        self.assertEqual(np.array(boxes1).shape, (1, 2, 4))
        self.assertEqual(np.array(boxes2).shape, (1, 3, 4))
        self.assertEqual(want.shape, (1, 2, 3))

        got = tb.axis_aligned_boxes.iou(boxes1, boxes2)

        np.testing.assert_array_almost_equal(got, want)

        # Test our box utils too.
        got_via_box_utils = self.compute_good_answer(
                np.array(boxes1), np.array(boxes2))
        np.testing.assert_almost_equal(got_via_box_utils, want)

    def testNan(self):
        boxes1 = np.array([
            [[1.0, 1.5, 3.0, 3.0],
             [0.0, 0.0, 0.0, 0.0],
            ]
        ])
        boxes2 = np.array([
            [[2.0, 2.5, 4.0, 4.0],
             [1.0, 1.5, 3.0, 3.0],
             [0.0, 0.0, 0.0, 0.0],
            ],
        ])

        want = np.array([
            [[0.5 / 5.5, 1.0, 0.0],
             [0.0,       0.0, np.NaN],
            ],
        ])

        self.assertEqual(boxes1.shape, (1, 2, 4))
        self.assertEqual(boxes2.shape, (1, 3, 4))
        self.assertEqual(want.shape, (1, 2, 3))

        got = tb.axis_aligned_boxes.iou(boxes1, boxes2)

    def testIntersectionRandom1d(self):
        boxes_1, boxes_2 = make_random_boxes( (10, 2, 1), (10, 5, 1) )
        assert boxes_1.shape == (10, 2, 2)
        want = self.compute_good_answer(boxes_1, boxes_2)
        got = tb.axis_aligned_boxes.iou(boxes_1, boxes_2)
        np.testing.assert_almost_equal(got, want)

    def testIntersectionRandom2d(self):
        BATCH_SIZE = 10
        boxes_1, boxes_2 = make_random_boxes( (BATCH_SIZE, 2000, 2), (BATCH_SIZE, 10, 2) )
        assert boxes_1.shape == (BATCH_SIZE, 2000, 4)
        print("Computing the right answer using for-loops (will take 10 sec)...")
        want = self.compute_good_answer(boxes_1, boxes_2)
        print("Computing the right answer using Tensors...")
        got = tb.axis_aligned_boxes.iou(boxes_1, boxes_2)

        found_nonzero = np.count_nonzero(got)
        print("done, found {} nonzero boxes!".format(found_nonzero))
        assert found_nonzero > 10000
        np.testing.assert_almost_equal(got, want)

    def testIntersectionRandom3d(self):
        boxes_1, boxes_2 = make_random_boxes( (17, 5, 3), (17, 11, 3) )
        assert boxes_1.shape == (17, 5, 6)
        want = self.compute_good_answer(boxes_1, boxes_2)
        got = tb.axis_aligned_boxes.iou(boxes_1, boxes_2)
        np.testing.assert_almost_equal(got, want)


class EvenlySpacedBoxesTest(unittest.TestCase):
    def testSimple(self):

        want = np.array(
                [[ -5.,   -7.5,   5.,    7.5],
                 [ -7.5,  -5.,    7.5,   5. ],
                 [ -5.,  104.5,   5.,  119.5],
                 [ -7.5, 107.,    7.5, 117. ],
                 [107.,   -7.5, 117.,    7.5],
                 [104.5,  -5.,  119.5,   5. ],
                 [107.,  104.5, 117.,  119.5],
                 [104.5, 107.,  119.5, 117. ]]
        )

        got = tb.axis_aligned_boxes.evenly_spaced(
            [ (2,2) ],
            [ [(10, 15), (15, 10)] ],
            (224, 224),
        )
        np.set_printoptions(threshold=sys.maxsize)
        print("Got: {}".format(got))

        np.testing.assert_array_almost_equal(got, want)


if __name__ == '__main__':
    unittest.main()

