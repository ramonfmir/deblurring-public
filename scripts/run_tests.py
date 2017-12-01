
import os
import sys
import unittest

base_dir = os.path.abspath(os.path.dirname(__file__))
modules = ("data_generator/", "cnn_denoiser/")

def run_tests():
	runner = unittest.TextTestRunner()

	failed_modules = []

	for module in modules:
		module_dir = os.path.join(base_dir, module)
		suite = unittest.loader.TestLoader().discover(module_dir + "tests/")
		exit_code = runner.run(suite).wasSuccessful()

		if not exit_code:
			 failed_modules.append(module)

	if failed_modules:
		print('The module(s) %s failed some tests' % ', '.join(failed_modules))
		return 1
	else:
		print('All tests ran successfully.')
		return 0

if __name__ == '__main__':
	sys.exit(run_tests())
