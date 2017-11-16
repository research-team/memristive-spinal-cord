#!/bin/sh

# PLEASE SET CORRECT PATHS before running

sudo apt-get update

# installing python and python libs
# sudo apt-get install python-all-dev python-numpy python-scipy python-matplotlib python-nose ipython cython --assume-yes
# installing cmake
sudo apt-get install cmake build-essential autoconf automake libncurses5-dev --assume-yes
# installing GNU Scientific Lib
sudo apt-get install gsl-bin libgsl2 libgsl0-dev libgsl0-dbg --assume-yes
# installing MPI
# sudo apt-get install libcr-dev mpich mpich-doc --assume-yes
sudo apt-get install openmpi-bin libopenmpi-dev --assume-yes
# installing LTDL
sudo apt-get install libtool libltdl7-dev --assume-yes
# installing Doxygen
sudo apt-get install doxygen --assume-yes
# installing pip3
sudo apt-get install python3-pip --assume-yes
# installing Readline
sudo apt-get install libreadline6 libreadline6-dev --assume-yes
# installing pkg-config
sudo apt-get install pkg-config cmake-data --assume-yes
# installing tkinter module
sudo apt-get install python3-tk --assume-yes

sudo apt-get install -y cmake build-essential autoconf automake libtool libltdl7-dev libreadline6-dev libncurses5-dev libgsl0-dev python-all-dev python-numpy python-scipy python-matplotlib python-nose ipython gsl-bin libgsl0-dev libgsl0-dbg cython

sudo -H pip3 install --upgrade pip
sudo -H pip3 install scipy nose matplotlib cython

NEST_VERSION="2.14.0"
NEST_NAME="nest-$NEST_VERSION"
NEST_PATH=/opt/nest
wget -c https://github.com/nest/nest-simulator/releases/download/v"$NEST_VERSION"/"$NEST_NAME".tar.gz -O $TMP_FOLDER/"$NEST_NAME".tar.gz
sudo mkdir -p $NEST_PATH
sudo chown -R "$USER" $NEST_PATH
mkdir -p $NEST_PATH/src
mkdir -p $NEST_PATH/build/$NEST_NAME
tar -xvzf $TMP_FOLDER/"$NEST_NAME".tar.gz -C $NEST_PATH/src
cd $NEST_PATH/build/$NEST_NAME
cmake -DCMAKE_INSTALL_PREFIX:PATH=$NEST_PATH/$NEST_NAME $NEST_PATH/src/$NEST_NAME -DPYTHON=3 -Dwith-mpi=ON -Dwith-python=3 -DPYTHON_EXECUTABLE=/usr/bin/python3.5 -DPYTHON_LIBRARY=/usr/lib/python3.5
make
make install
make installcheck
NEST_VARS=$NEST_PATH/$NEST_NAME/bin/nest_vars.sh
$NEST_VARS
source $NEST_VARS
if ! grep -q -F "source '$NEST_VARS'" ~/.profile ; then
  echo '# NEST env\n[ -f '$NEST_VARS' ] && source '$NEST_VARS >> ~/.profile
fi
