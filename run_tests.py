import os
import sys
import subprocess

base_dir = os.path.abspath(os.path.dirname(__file__))
modules = ("blurrer/", "deblurrer/")

def run_tests(compute_coverage):
	failed_modules = []
	for module in modules:
		module_dir = os.path.join(base_dir, module)

		if compute_coverage:
			result = subprocess.call(['coverage', 
									  'run', 
									  '--concurrency=eventlet', 
									  '--source=',  'tests'], cwd=module_dir)
		else:
			result = subprocess.call([sys.executable, 'tests'], cwd=module_dir)

		if result != 0:
			print('Tests failed: '.format(result))
			failed_modules.append(module)

	if compute_coverage:
        coverage_files = [app + '.coverage' for module in modules]
        subprocess.call(['coverage', 'combine'] + coverage_files, cwd=base_dir)

    if failed_modules:
    	print('The module(s) %s failed some tests' % ', '.join(failed_modules))
        return 1
    else:
    	print('All tests ran successfully.')
    	return 0


if __name__ == "__main__":
	compute_coverage = '--coverage' in sys.argv
	sys.exit(run_tests(compute_coverage))
