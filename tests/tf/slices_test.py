import sys
import numpy as np
import tensorflow as tf
import unittest

import tensorbank.tf as tb

class MultipleWithinStrideTest(unittest.TestCase):
    def testSimple(self):
        inp = tf.convert_to_tensor([])
        want = []
        np.testing.assert_array_almost_equal(
            tb.slice_within_stride(
                inp, 5),
            want)

    def testOneDimAll(self):
        inp = tf.convert_to_tensor([0,1,2,3,4,5,6,7])
        want = [0,1,2,3,4,5,6,7]
        np.testing.assert_array_almost_equal(
            tb.slice_within_stride(
                inp, 8),
            want)

    def testOneDimExtend(self):
        inp = tf.convert_to_tensor([0,1,2,3,4,5,6,7])
        want = [[0,1,2,3,4,5,6,7]]
        np.testing.assert_array_almost_equal(
            tb.slice_within_stride(
                inp, 8, keepdims=False),
            want)

    def testOneDimSome(self):
        inp = tf.convert_to_tensor([0,1,2,3,4,5,6,7])
        want = [0,1,2,3,4,5,6,7]
        np.testing.assert_array_almost_equal(
            tb.slice_within_stride(
                inp, 4),
            want)

    def testOneDimSomeExtend(self):
        inp = tf.convert_to_tensor([0,1,2,3,4,5,6,7])
        want = [[0,1,2,3],[4,5,6,7]]
        np.testing.assert_array_almost_equal(
            tb.slice_within_stride(
                inp, 4, keepdims=False),
            want)

    def testOneDimSomeStride(self):
        inp = tf.convert_to_tensor([0,1,2,3,4,5,6,7])
        want = [2,3,6,7]
        np.testing.assert_array_almost_equal(
            tb.slice_within_stride(
                inp, 4, 2),
            want)

    def testOneDimSomeExtendStride(self):
        inp = tf.convert_to_tensor([0,1,2,3,4,5,6,7])
        want = [[2,3],[6,7]]
        np.testing.assert_array_almost_equal(
            tb.slice_within_stride(
                inp, 4, 2, keepdims=False),
            want)

    def testOneDimSomeStrideEnd(self):
        inp = tf.convert_to_tensor([0,1,2,3,4,5,6,7])
        want = [2,6]
        np.testing.assert_array_almost_equal(
            tb.slice_within_stride(
                inp, 4, 2, 3),
            want)

    def testOneDimSomeExtendEnd(self):
        inp = tf.convert_to_tensor([0,1,2,3,4,5,6,7])
        want = [[2],[6]]
        np.testing.assert_array_almost_equal(
            tb.slice_within_stride(
                inp, 4, 2, 3, keepdims=False),
            want)

    def testOneDimSomeExtendEndMinus(self):
        inp = tf.convert_to_tensor([0,1,2,3,4,5,6,7])
        want = [[2],[6]]
        np.testing.assert_array_almost_equal(
            tb.slice_within_stride(
                inp, 4, 2, -1, keepdims=False),
            want)

if __name__ == '__main__':
    unittest.main()
