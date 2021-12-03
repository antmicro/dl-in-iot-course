#!/bin/bash

git clone --recursive https://github.com/tensorflow/tensorflow.git -b v2.7.0 tensorflow
cd tensorflow/
git am ../0001-tensorflow-lite-Added-limits-to-simple_memory_arena_.patch
cd -
