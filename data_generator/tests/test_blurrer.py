from __future__ import absolute_import

import unittest
import cv2
from functools import partial
import blurrer.blurring_experiment as bl
import blurrer.perspective_transformation as pt

class TestExample(unittest.TestCase):
	def test_averaging_blur(self):
		test_img = cv2.imread("blurrer/tests/license_plate.jpg")
		img = bl.averaging_blur(3, test_img)
		self.assertNotEqual(test_img.tolist(), img.tolist())

	def test_gaussian_blur(self):
		test_img = cv2.imread("blurrer/tests/license_plate.jpg")
		img = bl.gaussian_blur(3, 2, test_img)
		self.assertNotEqual(test_img.tolist(), img.tolist())

	def test_median_blur(self):
		test_img = cv2.imread("blurrer/tests/license_plate.jpg")
		img = bl.median_blur(3, test_img)
		self.assertNotEqual(test_img.tolist(), img.tolist())

	def test_bilateral_blur(self):
		test_img = cv2.imread("blurrer/tests/license_plate.jpg")
		img = bl.bilateral_blur(3, 3, test_img)
		self.assertNotEqual(test_img.tolist(), img.tolist())

	def test_motion_blur(self):
		test_img = cv2.imread("blurrer/tests/license_plate.jpg")
		img = bl.motion_blur(3, 'H', 2, test_img)
		self.assertNotEqual(test_img.tolist(), img.tolist())

	def test_apply_blurs_randomly(self):
		test_img = cv2.imread("blurrer/tests/license_plate.jpg")
		averaging_b = partial(bl.averaging_blur, 3)
		median_b    = partial(bl.median_blur, 3)
		gaussian_b  = partial(bl.gaussian_blur, 3, 2)
		bilateral_b = partial(bl.bilateral_blur, 3, 3)
		motion_b    = partial(bl.motion_blur, 3, 2, 'H')

		blurs = [averaging_b, median_b, gaussian_b, bilateral_b, motion_b]

		img = bl.apply_blurs_randomly(blurs, test_img)
		self.assertNotEqual(test_img.tolist(), img.tolist())
