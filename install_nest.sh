#!/bin/sh

# PLEASE SET CORRECT PATHS before running

sudo apt-get update

# installing cmake
sudo apt-get install cmake build-essential --assume-yes
# installing GNU Scientific Lib
sudo apt-get install libgsl2 libgsl0-dev --assume-yes
# installing MPI
sudo apt-get install libcr-dev mpich mpich-doc --assume-yes
# installing LTDL
sudo apt-get install libtool --assume-yes
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

pip3 install scipy nose matplotlib cython

NEST_VERSION="2.12.0"
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
bash $NEST_VARS
if ! grep -q -F "source '$NEST_VARS'" ~/.profile ; then
  echo '# NEST env\n[ -f '$NEST_VARS' ] && source '$NEST_VARS >> ~/.profile
fi
