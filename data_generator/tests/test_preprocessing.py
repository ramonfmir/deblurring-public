from __future__ import absolute_import
from shutil import copyfile
from copy import deepcopy

import sys
import os

import unittest
import cv2
import data_generator.preprocessing.colour_normalize as cn
import data_generator.preprocessing.histogram_equalisation as he
import data_generator.preprocessing.kmeans as km
import data_generator.preprocessing.permuter as pm

path = "data_generator/tests/"
test_img_name = "license_plate.jpg"
test_img = cv2.imread(path + test_img_name, 0)

class TestPreprocessing(unittest.TestCase):
	def test_colour_normalize(self):
		dir_to_normalize = path + "tmp/"
		os.mkdir(dir_to_normalize)
		copyfile(path + test_img_name, dir_to_normalize + test_img_name)
		cn.normalize(dir_to_normalize)
		self.assertFalse(cv2.imread(dir_to_normalize + test_img_name).tolist() == None)
		os.remove(dir_to_normalize + test_img_name)
		os.rmdir(dir_to_normalize)

	def test_histogram_equalisation(self):
		self.assertNotEqual(test_img.tolist(), he.clahe(test_img, 3, 5).tolist())

	def test_kmeans(self):
		copyfile(path + test_img_name, path + "tmp.jpg")
		km.knn_colour(path + test_img_name, path + "tmp.jpg", 2)
		result = cv2.imread(path + "tmp.jpg")
		self.assertNotEqual(test_img.tolist(), result.tolist())
		os.remove(path + "tmp.jpg")

	def test_permuter(self):
		dir_to_permute = path + "tmp/"
		os.mkdir(dir_to_permute)
		copyfile(path + test_img_name, dir_to_permute + test_img_name)
		pm.permute(dir_to_permute + test_img_name)
		result = cv2.imread(dir_to_permute + "license_plate_0.jpg")
		self.assertNotEqual(test_img.tolist(), result.tolist())
		os.remove(dir_to_permute + test_img_name)
		for i in range(50):
			os.remove(dir_to_permute + "license_plate_" + str(i) + ".jpg")
		os.rmdir(dir_to_permute)
