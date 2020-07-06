import sys
import numpy as np
import tensorflow as tf
import unittest

import tensorbank.tf as tb

class PairwiseL2NormTest(unittest.TestCase):
    def testSimple(self):
        points1 = np.array([[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]])
        points2 = np.array([[[1.1, 3.1, 4.1], [1.2, 3.2, 4.2], [1.3, 3.3, 4.3]]])
        self.assertEqual(points1.shape, (1, 2, 3))
        self.assertEqual(points2.shape, (1, 3, 3))

        want = np.array([
            [[2.43,  2.92,  3.47],
             [0.83,  0.72,  0.67]]
        ])

        got = tb.points.pairwise_l2_distance(points1, points2, sqrt=False)
        np.testing.assert_array_almost_equal(got, want)

        got_sqrt = tb.points.pairwise_l2_distance(points1, points2)
        np.testing.assert_array_almost_equal(got_sqrt, tf.sqrt(want))

        got_sqrt = tb.points.pairwise_l2_distance(points1, points2, sqrt=True)
        np.testing.assert_array_almost_equal(got_sqrt, tf.sqrt(want))


class PairwiseL1NormTest(unittest.TestCase):
    def testSimple(self):
        points1 = np.array([[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]])
        points2 = np.array([[[1.1, 3.1, 4.1], [1.2, 3.2, 4.2], [1.3, 3.3, 4.3]]])
        self.assertEqual(points1.shape, (1, 2, 3))
        self.assertEqual(points2.shape, (1, 3, 3))

        want = np.array([
            [[2.3, 2.6, 2.9],
             [1.1, 1.2, 1.3]]
        ])

        got = tb.points.pairwise_l1_distance(points1, points2)

        np.testing.assert_array_almost_equal(got, want)


class PairwiseLInfNormTest(unittest.TestCase):
    def testSimple(self):
        points1 = np.array([[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]])
        points2 = np.array([[[1.1, 3.1, 4.1], [1.2, 3.2, 4.2], [1.3, 3.3, 4.3]]])
        self.assertEqual(points1.shape, (1, 2, 3))
        self.assertEqual(points2.shape, (1, 3, 3))

        want = np.array([
            [[1.1, 1.2, 1.3],
             [0.9, 0.8, 0.7]]
        ])

        got = tb.points.pairwise_l_inf_distance(points1, points2)

        np.testing.assert_array_almost_equal(got, want)


if __name__ == '__main__':
    unittest.main()


