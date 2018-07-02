# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from unittest import TestCase

import tensorflow as tf
from jpeg2dct.tensorflow import decode


class TestLoad(TestCase):
    def setUp(self):
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.jpeg_file = os.path.join(cur_dir, '..', 'data', 'DCT_16_16.jpg')

        self.jpeg_file_411 = os.path.join(cur_dir, '..', 'data', 'DCT_16_16_411.jpg')
        self.jpeg_file_420 = os.path.join(cur_dir, '..', 'data', 'DCT_16_16_420.jpg')
        self.jpeg_file_422 = os.path.join(cur_dir, '..', 'data', 'DCT_16_16_422.jpg')
        self.jpeg_file_440 = os.path.join(cur_dir, '..', 'data', 'DCT_16_16_440.jpg')
        self.jpeg_file_444 = os.path.join(cur_dir, '..', 'data', 'DCT_16_16_444.jpg')

        self.sess = tf.Session()

    def tearDown(self):
        self.sess.close()

    @staticmethod
    def bytess_helper(jpeg_fname):
        jpegbytes = tf.read_file(jpeg_fname)
        return jpegbytes

    def test_decode(self):
        dct_y_tf, dct_c_tf, dct_r_tf = decode(self.bytess_helper(self.jpeg_file))
        self.assertEqual(len(dct_y_tf.shape), 3, "shall be a tensore of 3 dimensions")
        self.assertEqual(dct_y_tf.shape[-1], 64, "last dimension shall be 64")

        with self.sess.as_default():
            dct_y = dct_y_tf.eval()
            dct_c = dct_c_tf.eval()
            dct_r = dct_r_tf.eval()
            self.assertEqual(dct_y.shape, (205, 205, 64), "wrong dct shape")
            self.assertEqual(dct_c.shape, (103, 103, 64), "wrong dct shape")
            self.assertEqual(dct_r.shape, (103, 103, 64), "wrong dct shape")

        dct_y_tf_un, dct_c_tf_un, dct_r_tf_un = decode(self.bytess_helper(self.jpeg_file), normalized=False)
        self.assertEqual(len(dct_y_tf_un.shape), 3, "shall be a tensore of 3 dimensions")
        self.assertEqual(dct_y_tf_un.shape[-1], 64, "last dimension shall be 64")

        with self.sess.as_default():
            dct_y_nonormalized = dct_y_tf_un.eval()
            dct_c_nonormalized = dct_c_tf_un.eval()
            dct_r_nonormalized = dct_r_tf_un.eval()
            self.assertEqual(dct_y_nonormalized.shape, (205, 205, 64), "wrong dct shape")
            self.assertEqual(dct_c_nonormalized.shape, (103, 103, 64), "wrong dct shape")
            self.assertEqual(dct_r_nonormalized.shape, (103, 103, 64), "wrong dct shape")


        normalized_range = dct_y.min(), dct_y.max()
        unnormalized_range = dct_y_nonormalized.min(), dct_y_nonormalized.max()
        self.assertTrue(unnormalized_range[0] >= normalized_range[0] and
                            unnormalized_range[1] <= normalized_range[1],
                            "normalized shall produce large range of values")

        normalized_range = dct_c.min(), dct_c.max()
        unnormalized_range = dct_c_nonormalized.min(), dct_c_nonormalized.max()
        self.assertTrue(unnormalized_range[0] >= normalized_range[0] and
                            unnormalized_range[1] <= normalized_range[1],
                            "normalized shall produce large range of values")

        dct_y_tf_onechannel = decode(self.bytess_helper(self.jpeg_file), channels=1)[0]
        with self.sess.as_default():
            self.assertEqual(dct_y_tf_onechannel.eval().shape, (205, 205, 64), "wrong dct shape")

    def test_transcoding(self):
        with self.sess.as_default():
            dct_y, dct_c, dct_r = decode(self.bytess_helper(self.jpeg_file_411))
            self.assertEqual(dct_y.eval().shape, (50, 75, 64), "wrong dct shape")
            self.assertEqual(dct_c.eval().shape, (25, 38, 64), "wrong dct shape")
            self.assertEqual(dct_r.eval().shape, (25, 38, 64), "wrong dct shape")

            dct_y, dct_c, dct_r = decode(self.bytess_helper(self.jpeg_file_420))
            self.assertEqual(dct_y.eval().shape, (50, 75, 64), "wrong dct shape")
            self.assertEqual(dct_c.eval().shape, (25, 38, 64), "wrong dct shape")
            self.assertEqual(dct_r.eval().shape, (25, 38, 64), "wrong dct shape")

            dct_y, dct_c, dct_r = decode(self.bytess_helper(self.jpeg_file_422))
            self.assertEqual(dct_y.eval().shape, (50, 75, 64), "wrong dct shape")
            self.assertEqual(dct_c.eval().shape, (25, 38, 64), "wrong dct shape")
            self.assertEqual(dct_r.eval().shape, (25, 38, 64), "wrong dct shape")

            dct_y, dct_c, dct_r = decode(self.bytess_helper(self.jpeg_file_440))
            self.assertEqual(dct_y.eval().shape, (50, 75, 64), "wrong dct shape")
            self.assertEqual(dct_c.eval().shape, (25, 38, 64), "wrong dct shape")
            self.assertEqual(dct_r.eval().shape, (25, 38, 64), "wrong dct shape")

            dct_y, dct_c, dct_r = decode(self.bytess_helper(self.jpeg_file_444))
            self.assertEqual(dct_y.eval().shape, (50, 75, 64), "wrong dct shape")
            self.assertEqual(dct_c.eval().shape, (25, 38, 64), "wrong dct shape")
            self.assertEqual(dct_r.eval().shape, (25, 38, 64), "wrong dct shape")