#!/bin/bash

git clone --recursive https://github.com/google/CFU-Playground.git

cd CFU-Playground

git checkout 0612528704a5d177235b3321d3ada2f2946c5e30

git am ../*.patch

cd -
