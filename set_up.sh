#! /bin/bash

# Shell script set_up.sh
# Set up opencv-contrib

#yes|yum install python34 python34-setuptools wget
#yes|easy_install-3.4 pip

wget https://pypi.python.org/packages/98/2c/cc7f2268ef5a276153e320f070db5f96afded0d99e9b957558c68d6abbec/opencv_contrib_python-3.4.0.12-cp27-cp27m-manylinux1_x86_64.whl#md5=879fafbed3e2da442e44d373976488f2
pip install opencv_contrib_python-3.4.0.12-cp27-cp27m-manylinux1_x86_64.whl

pip install numpy scipy

