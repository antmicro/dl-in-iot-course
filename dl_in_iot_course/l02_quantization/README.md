# Optimization Algorithms - Quantization

This aim of this laboratory is to demonstrate the quantization on deep learning models using TensorFlow Lite.

The model used for quantization tasks is a MobileNetV2-based model for classifying dog and cat breeds.
The dataset used for training, evaluating and quantizing the models is [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).

The model for below tasks was trained using [model_training script](model_training.py).
The trained model is available under [models directory](../../models) as `pet-dataset-tensorflow.h5`.

All tasks should be implemented in [quantization_experiments script](quantization_experiments.py).
The current code in this script implements a `ModelTester` class that runs evaluation and performance measurements for the model.
It requires implementing methods for:

* preprocessing inputs (`preprocess_input`) - while the PetDataset object performs image loading and preprocessing, the trained and compiled models quite often require additional steps before inferring data, such as:

    * Proper data allocation,
    * Data quantization (for INT8 quantization).
* postprocessing outputs (`postprocess_outputs`) - similarly as in preprocessing inputs, sometimes additional steps need to be made to get the proper results.
* running inference (`run_inference`) - this method should **only** run inference method, all data pre- and postprocessing aspects (including allocation, scaling) should be done in the above-mentioned methods.
* preparing model (`prepare_model`) - trained and compiled models may have different loading methods.
* optimizing model (`optimize_model`) - this is the place where model compilation and optimization should take place.

## Tasks

The tasks below can be implemented using either `tf.lite.Interpreter` "legacy" runtime, or new `ai_edge_litert.interpreter.Interpreter` runtime.
Both variants of `Interpreter` use the same API, so in general it is a straightforward replacement.
The changes appear in the format of the Keras's models.
In `./models` directory of the project there are two "native" models:

* `pet-dataset-tensorflow.h5` - trained using TensorFlow 2.19 (should be supported by versions 2.16 onwards)
* `pet-dataset-tensorflow-2-13.h5` - trained using TensorFlow 2.11 (should be support by versions up to 2.15)

In case of any problems please notify the tutor.

Tasks:

* `[2pt]` Learn about your hardware - it is important to know the target platform you will run your models on.
  It will become especially important in the future tasks.
  For now, let's determine the following aspects (write them in the summary):

  * The CPU present in the platform: `lscpu` or `cat /proc/cpuinfo`
  * The RAM present in the platform, e.g. `lshw -c memory` (many other approaches are available)
* Go over the [quantization_experiments script](quantization_experiments.py) and check what it does (go to methods from other modules to get the better understanding on how the solution works).
* `[1pt]` In `NativeModel` class, in the `prepare_model` method, add printing summary of the model (there is a dedicated method for this in TensorFlow models) - check out the number of parameters in the model.
* `[4pt]` Finish the `FP32Model` class:

    * in `optimize_model`, load the trained model, create a `tf.lite.TFLiteConverter` object from the model, convert it to the TFLite model without any optimizations and save results to the file under `self.modelpath` path.
    * in `prepare_model`, create a `tf.lite.Interpreter` (or `ai_edge_litert.interpreter.Interpreter`) for the model saved in `self.modelpath` path.
      I'd suggest setting a `num_threads` parameter here to the number of threads available in the CPU to significantly boost the inference process.
      You can use e.g. `multiprocessing` module to collect number of available cores.
      Remember to allocate tensors (there is a method for it).
    * in `run_inference` method, invoke the model.
    * in `postprocess_outputs`, implement the method for getting the output tensor (check out `preprocess_input` method for hints on how to do it).

* `[4pt]` Finish the `INT8Model` class:

    * In `optimize_model`, optimize a model to work in full `int8` mode:

        * use `tf.lite.Optimize.DEFAULT` optimization,
        * use `calibration_dataset_generator` as `converter.representative_dataset`,
        * use `tf.int8` as inference input and output type.
        * in general, set following members of the converter object: `optimizations`, `representative_dataset`, `target_spec.supported_ops`, `inference_input_type` and `inference_output_type`
    * Implement `prepare_model`, `run_inference` following the same flow as above
    * Implement `preprocess_input` and `postprocess_outputs` methods:

        * remember to quantize the inputs and dequantize the outputs (`scale` and `zero_point` parameters are present in `self.model.get_input_details()[0]['quantization']` field, respectively)!

* `[2pt]` Finish the `ImbalancedINT8Model` class:

    * Implement `optimize_model` method, where the `calibration_dataset_generator` will take all examples for objects with 5 class and use them for calibration:

        * Use `self.dataset.dataX` and `self.dataset.dataY` to extract all inputs for a particular class.
        * Remember to use self.dataset.prepare_input_sample method.

* The main script can be executed like so:

  ```
  python3 -m dl_in_iot_course.l02_quantization.quantization_experiments \
        --model-path models/pet-dataset-tensorflow.h5 \
        --dataset-root build/pet-dataset/ \
        --results-path build/results --tasks <space-separated-list-of-tasks-to-run>
  ```

  Where `--tasks` is a space-separated list of models to run.
  Possible variants here are:

  * `all` - runs all models (will fail if any model is not fully implemented)
  * `native` - run model using native framework (default)
  * `tflite-fp32` - runs model using TensorFlow Lite (or LiteRT) with FP32 precision
  * `tflite-int8-<percentage>` - quantizes model using `<percentage>` of the training dataset for calibration (0.01, 0.08, 0.3, 0.8) values are available
  * `tflite-int8-all` - runs all variants of quantized models with balanced calibration datasets
  * `tflite-imbint8` - runs quantization and evaluation of model using imbalanced dataset

  The above can be provided as a list to run only selected variants.

  `NOTE:` To download the dataset, add `--download-dataset` flag.

  `NOTE:` If the evaluation takes too long, reduce the test dataset size by setting `--test-dataset-fraction` to some lower value, but inform about this in the Summary note.

  `NOTE:` Due to Python's garbage collection, in some cases it'll be recommended on some platforms to run each model/task separately.

  In the `build/results` directory, the script will create:

    * `<prefix>-metrics.md` file - contains basic metrics, such as accuracy, precision, sensitivity or G-Mean, along with inference time
    * `<prefix>-confusion-matrix.png` file - contains visualization of confusion matrix for the model evaluation.
  Those files will be created for:

    * `native` - the model running in TensorFlow framework,
    * `tflite-fp32` - the model running in TFLite runtime with FP32 precision,
    * `tflite-int8-<calibsize>` - the model running in TFLite runtime with INT8 precision calibrated with `<calibsize>` fraction of training dataset,
    * `tflite-imbint8` - the model running in TFLite runtime with INT8 precision calibrated with samples for several classes.

* Write a small summary for experiments containing:

    * `[2pt]` Include information regarding CPU (`lscpu` or `cat /proc/cpuinfo` results) and information about available RAM and answer following questions:
        * What is `CPU(s)` count - is it number of CPUs, cores or something else?
        * What are Flags in the results and what can they tell us about the platform in terms of Edge AI deployment?
    * `[1pt]` Number of parameters in the model (in total),
    * `[1pt]` Size of FP32 TFLite model, and size of the INT8 model - compare the size reduction (check file sizes),
    * For each experiment include:

        * The computed metrics,
        * Confusion matrix.
    * Answers for the questions:

        * `[1pt]` How does the TFLite FP32 model perform in comparison to native model (both performance- and quality-wise)?
        * `[1pt]` How does the best INT8 model perform in comparison to the TFLite FP32 model (both performance- and quality-wise)?
        * `[1pt]` Is there any specific trend observable in the quality of INT8 models based on calibration dataset size?
        * `[1pt]` How does the model calibrated with samples of only one class perform in comparison to other INT8 models?

  The summary should be put in the project's `summaries` directory - follow the README.md in this directory for details.

Additional factors:

* `[2pt]` Git history quality

`NOTE:` There is no need to include the models in the repository.

`NOTE:` The performance of a given runtime depends heavily on chosen hardware.

`NOTE:` Confusion matrix shows clearly if there are any issues with the optimized model.
If the confusion matrix is almost random (with no significantly higher values along the diagonal) - there are possible issues with the model, usually within preprocessing step (make sure to use `scale`, `zero_point` parameters and to convert the input data to `int8` type).

## Resources

Most of the documentation regarding quantization in TensorFlow Lite can be found in:

* [TFLite Converter documentation](https://www.tensorflow.org/lite/convert)
* [TFLite Interpreter documentation](https://www.tensorflow.org/lite/guide/inference)
* [TFLite Post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)
* [TFLite Quantization specification](https://www.tensorflow.org/lite/performance/quantization_spec)
