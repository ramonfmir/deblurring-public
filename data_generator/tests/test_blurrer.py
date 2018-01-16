from __future__ import absolute_import
from shutil import copyfile
from copy import deepcopy

import sys
import os

import unittest
import cv2
import data_generator.blurring.corrupter as corrupter

path = "data_generator/tests/"
test_img_name = "license_plate.jpg"
test_img = cv2.imread(path + test_img_name)

class TestBlurrer(unittest.TestCase):
	def test_corrupter(self):
		_, test_img_main = corrupter.blur_main(test_img)
		_, test_img_motion = corrupter.blur_vertical(test_img)
		_, test_img_brightness = corrupter.blur_brightness(test_img)
		_, test_img_pixelation = corrupter.blur_pixelation(test_img)
		_, test_img_mild = corrupter.blur_mild(test_img)

		self.assertNotEqual(test_img.tolist(), test_img_main.tolist())
		self.assertNotEqual(test_img.tolist(), test_img_motion.tolist())
		self.assertNotEqual(test_img.tolist(), test_img_brightness.tolist())
		self.assertNotEqual(test_img.tolist(), test_img_pixelation.tolist())
		self.assertNotEqual(test_img.tolist(), test_img_mild.tolist())