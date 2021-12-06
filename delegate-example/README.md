# TensorFlow Lite external delegate example

This directory contains a sample delegate using Eigen library for linear algebra operations.

## Building the delegate

First of all, clone the TensorFlow v2.7.0 repository with a patch using `clone-tensorflow.sh` script:

```
./clone-tensorflow.sh
```

Later, run CMake to build the delegate:

```
mkdir build
cd build
cmake ..
make -j`nproc`
```

This will build a delegate libeigen-delegate.so that can be used from TensorFlow Lite Python library.
