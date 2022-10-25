# Optimization of Deep Learning applications for IoT devices - Course tasks

Copyright (c) 2021-2022 [Antmicro](https://www.antmicro.com)

This repository contains tasks for laboratories for the "Optimization of Deep Learning applications for IoT devices" course.

## Course classes

Each of the `l<number>_<topic>` directories in the `dl_in_iot_course` module contain a separate README with the list of tasks.
Please follow the links to go to the list of tasks:

* [Lab 02 - Quantization with TensorFlow Lite](dl_in_iot_course/l02_quantization)
* [Lab 03 - Pruning and clustering with TensorFlow Model Optimization Toolkit](dl_in_iot_course/l03_pruning_clustering)
* [Lab 04 - Introduction to Apache TVM](dl_in_iot_course/l04_tvm)
* [Lab 05 - Implementing a TensorFlow Lite delegate](dl_in_iot_course/l05_tflite_delegate)
* [Lab 06 - Fine-tuning of model and operations in Apache TVM](dl_in_iot_course/l06_tvm_fine_tuning)

## Cloning the repository

`NOTE:` [Git LFS tool](https://git-lfs.github.com/) is required to pull the large files, such as models.
Install it before cloning the repository.

To clone the repository with all models, run:

```
git clone --recursive https://github.com/antmicro/dl-in-iot-course.git
cd dl-in-iot-course/models
git lfs pull
cd ..
```

## Environment preparation

To provide a consistent environment for running tasks, the [Sylabs Singularity](https://sylabs.io/singularity/) image definitions are available in the [environments directory](environments/).

To get started with the Singularity environment, check the [Quick Start guide](https://sylabs.io/guides/3.8/user-guide/quick_start.html) for installation and running steps.

To build the SIF files from image definitions, run:

```
cd environments/
mkdir tmp
env SINGULARITY_TMPDIR=(pwd)/tmp sudo -E singularity build development-environment.sif development-environment.def
cd ..
```

`NOTE:` Use `development-environment-gpu.def` definition for the GPU-enabled version of the image.

To start working in the container, run:

```
singularity shell environments/development-environment.sif
```

To use the GPU-enabled container (only for NVIDIA with CUDA), run:

```
singularity shell --nv environments/development-environment-gpu.sif
```

Singularity by default enables using GUI, makes all of the devices available from the container, and mounts the /home directory by default.

## Running the executable scripts in the repository

In order to handle submodules easily, all of the executable scripts should be started from the root of the repository, i.e.:

```
python3 -m dl_in_iot_course.l02_quantization.quantization_experiments -h
```
