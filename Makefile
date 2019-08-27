SRC = face_recognition_ros/src face_recognition_ros/nodes/*

.PHONY: clean clean-pyc clean-build 

clean: clean-build clean-pyc ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

format:
	black $(SRC)

lint: ## check style with flake8
	flake8 $(SRC)

#dist: clean ## builds source and wheel package
#	python setup.py sdist
#	python setup.py bdist_wheel
#	ls -l dist

#install: clean ## install the package to the active Python's site-packages
#	python setup.py install

build: clean
	cd ~/catkin_ws; \
	catkin_make
