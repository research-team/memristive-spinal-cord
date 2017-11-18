#!/bin/sh

# PLEASE SET CORRECT PATHS before running

# This script tested on the clean installed Ubuntu Server 16.04.3
# To install the NEST just run install_nest.sh without sudo (to avoid mpi warnings which will interrupt a testing phase)

echo "Hello! This scipt will install NEST Simulator (2.12.0 or 2.14.0) with Python (2 or 3)"
echo "Which version of NEST do you want to install? (default 2.12.0):"
CORRECT=false
while ! $CORRECT ; do
  read NEST_VERSION
  if test "$NEST_VERSION" = "2.12.0" || test "$NEST_VERSION" = "2.14.0" ; then
    CORRECT=true
  elif test "$NEST_VERSION" = "" || test "$NEST_VERSION" = "12" ; then
    NEST_VERSION="2.12.0"
    CORRECT=true
  elif test "$NEST_VERSION" = "14" ; then
    NEST_VERSION="2.12.0"
    CORRECT=true
  else
    echo "Please, choose 2.12.0, or 2.14.0, or just press ENTER to install 2.12.0"
  fi
done
echo "$NEST_VERSION has been chosen"
echo ""
echo "Which directory will be user to install NEST? (default /opt/nest):"
read NEST_PATH
if test "$NEST_PATH" = "" ; then
  NEST_PATH="/opt/nest"
fi
echo "NEST will be installed in $NEST_PATH"

echo ""
echo "Which Python version (2 or 3) will be used? (default 3):"
read PYTHON_VERSION
CORRECT=false
while ! $CORRECT ; do
  if test "$PYTHON_VERSION" = "2" || test "$PYTHON_VERSION" = "3" ; then
    CORRECT=true
  elif test "$PYTHON_VERSION" = "" ; then
    PYTHON_VERSION="3"
    CORRECT=true
  else
    echo "Please, choose 2, or 3, or just press ENTER to use Python 3"
  fi
done
echo "Python $PYTHON_VERSION has been chosen"

if test "`which python$PYTHON_VERSION`" = "" ; then
  echo "Python $PYTHON_VERSION is not found and will be installed"
  read WAITING
  if test "$PYTHON_VERSION" = "2" ; then
    sudo apt-get install python-minimal --assume-yes
  elif test "$PYTHON_VERSION" = "3" ; then
    sudo apt-get install python3-minimal --assume-yes
  fi
fi
PYTHON_EXECUTABLE=`which python$PYTHON_VERSION`
echo "Python $PYTHON_VERSION executable: $PYTHON_EXECUTABLE"
echo "Press ENTER to install NEST $NEST_VERSION with Python$PYTHON_VERSION into $NEST_PATH"
read WAITING

sudo apt-get update

# installing python and python libs
sudo apt-get install python-all-dev python-numpy python-scipy python-matplotlib python-nose ipython cython --assume-yes
# installing cmake
sudo apt-get install cmake build-essential autoconf automake libncurses5-dev --assume-yes
# installing GNU Scientific Lib
sudo apt-get install gsl-bin libgsl2 libgsl0-dev libgsl0-dbg --assume-yes
# installing MPI
sudo apt-get install openmpi-bin libopenmpi-dev --assume-yes
# installing LTDL
sudo apt-get install libtool libltdl7-dev --assume-yes
# installing Doxygen
sudo apt-get install doxygen --assume-yes
# installing pip and others with pip
if test "$PYTHON_VERSION" = "2" ; then
  sudo apt-get install pip --assume-yes
  sudo -H pip install --upgrade pip
  sudo -H pip install scipy nose matplotlib cython
elif test "$PYTHON_VERSION" = "3" ; then
  sudo apt-get install python3-pip --assume-yes
  sudo -H pip3 install --upgrade pip
  sudo -H pip3 install scipy nose matplotlib cython
fi
# installing Readline
sudo apt-get install libreadline6 libreadline6-dev --assume-yes
# installing pkg-config
sudo apt-get install pkg-config cmake-data --assume-yes
# installing tkinter module
sudo apt-get install python3-tk --assume-yes

# NEST_VERSION="2.12.0"
NEST_NAME="nest-$NEST_VERSION"
# NEST_PATH=/opt/nest
sudo wget -c https://github.com/nest/nest-simulator/releases/download/v"$NEST_VERSION"/"$NEST_NAME".tar.gz -O $TMP_FOLDER/"$NEST_NAME".tar.gz
sudo mkdir -p $NEST_PATH
sudo chown -R "$USER" $NEST_PATH
mkdir -p $NEST_PATH/src
mkdir -p $NEST_PATH/build/$NEST_NAME
tar -xvzf $TMP_FOLDER/"$NEST_NAME".tar.gz -C $NEST_PATH/src
cd $NEST_PATH/build/$NEST_NAME
cmake -DCMAKE_INSTALL_PREFIX:PATH=$NEST_PATH/$NEST_NAME $NEST_PATH/src/$NEST_NAME -Dwith-mpi=ON -Dwith-python=$PYTHON_VERSION -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE # -DPYTHON_LIBRARY=/usr/lib/python3.5
make
make install

# the test_quantal_stp_synapse.py uses old numpy library, so NEST 2.12.0 fails the test
# by this reason these t_plot and t_tot values have to be int, not float
if test "$NEST_VERSION" = "2.12.0" ; then
  TEST_PATH=$NEST_PATH/$NEST_NAME/lib/python3.5/site-packages/nest/tests/test_quantal_stp_synapse.py
  sudo sed -ie 's/t_plot = 1000./t_plot = 1000/' $TEST_PATH
  sudo sed -ie 's/t_tot = 1500./t_tot = 1500/' $TEST_PATH
fi
make installcheck
NEST_VARS=$NEST_PATH/$NEST_NAME/bin/nest_vars.sh
$NEST_VARS
bash $NEST_VARS
if ! grep -q -F "source '$NEST_VARS'" ~/.profile ; then
  echo '# NEST env\n[ -f '$NEST_VARS' ] && source '$NEST_VARS >> ~/.profile
fi
