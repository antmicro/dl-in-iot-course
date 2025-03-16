# Optimization of Deep Learning applications for IoT devices - Course tasks

Copyright (c) 2021-2025 [Antmicro](https://www.antmicro.com)

This repository contains tasks for laboratories for the "Optimization of Neural Network applications for IoT devices" course.

## Course classes

Each of the `l<number>_<topic>` directories in the `dl_in_iot_course` module contain a separate README with the list of tasks.
Please follow the links to go to the list of tasks:

* [Lab 02 - Quantization with TensorFlow Lite](dl_in_iot_course/l02_quantization)
* [Lab 03 - Pruning and clustering with TensorFlow Model Optimization Toolkit](dl_in_iot_course/l03_pruning_clustering)
* [Lab 04 - Introduction to Apache TVM](dl_in_iot_course/l04_tvm)
* [Lab 05 - Implementing a TensorFlow Lite delegate](dl_in_iot_course/l05_tflite_delegate)
* [Lab 06 - Fine-tuning of model and operations in Apache TVM](dl_in_iot_course/l06_tvm_fine_tuning)
* [Lab 07 - Accelerating ML models on FPGAs with TFLite Micro and CFU Playground](cfu-playground)

Each README provides instructions on:

* What to do in the tasks
* How to run experiments
* How to prepare the summary

## Cloning the repository

`NOTE:` [Git LFS tool](https://git-lfs.github.com/) is required to pull large files, such as models.
Install it before cloning the repository.

To clone the repository with all models, run:

```
git clone --recursive https://github.com/antmicro/dl-in-iot-course.git
cd dl-in-iot-course/models
git lfs pull
cd ..
```

## Environment preparation


### Using Docker image

The recommended approach is to use a Docker image that provides all dependencies necessary for running tasks from the project.

The definition for the Docker image is located in [`environments` directory](./environments/Dockerfile).

To pull the built image, run:

```bash
docker pull ghcr.io/antmicro/dl-in-iot-course:latest
```

To run it and automatically include the current workspace directory, you can run:

```bash
docker run -it --rm -v $(pwd):$(pwd) -w $(pwd) ghcr.io/antmicro/dl-in-iot-course:latest /bin/bash
```

From this point, you can run tasks for the project.

### Using virtual environment

The dependencies for tasks are provided in the `requirements.txt` file.
To install them, first create a virtual environment with `venv` in project's directory:

```bash
python3 -m venv .venv
```

After this, activate the environment:

```bash
source ./.venv/bin/activate
```

And proceed with installing necessary dependencies:

```bash
pip3 install -r requirements.txt
```

## Running the executable scripts in the repository

In order to handle Python modules for the project easily, all of the executable scripts should be started from the root of the repository, i.e.:

```
python3 -m dl_in_iot_course.l02_quantization.quantization_experiments -h
```
