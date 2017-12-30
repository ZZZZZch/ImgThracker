#! /bin/bash

# Shell script set_up.sh
# Set up opencv-contrib

yes|yum install python34 python34-setuptools wget
yes|easy_install-3.4 pip

wget https://pypi.python.org/packages/5e/01/d482c01255bd2f742637bc83a93384d17bb595e5065b43e7665414850897/opencv_contrib_python-3.4.0.12-cp34-cp34m-manylinux1_x86_64.whl#md5=ae97bd49db3c28cc76e0d2c0ddd267e2
pip3 install opencv-python opencv_contrib_python-3.4.0.12-cp34-cp34m-manylinux1_x86_64.whl

pip3 install numpy scipy

