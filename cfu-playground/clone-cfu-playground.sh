#!/bin/bash

git clone --recursive https://github.com/google/CFU-Playground.git

cd CFU-Playground

git checkout 328263b92abc8b607e02812db4d386b2759c2041

git am ../*.patch

cd -
