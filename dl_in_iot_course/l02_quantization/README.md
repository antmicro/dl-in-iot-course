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

* Go over the [quantization_experiments script](quantization_experiments.py) and check what it does (go to methods from other modules to get the better understanding on how the solution works).
* `[1pt]` In `NativeModel` class, in the `prepare_model` method, add printing summary of the model (there is a dedicated method for this in TensorFlow models) - check out the number of parameters in the model.
* `[4pt]` Finish the `FP32Model` class:

    * in `optimize_model`, load the trained model, create a `tf.lite.TFLiteConverter` object from the model, convert it to the TFLite model without any optimizations and save results to the file under `self.modelpath` path.
    * in `prepare_model`, create a `tf.lite.Interpreter` for the model saved in `self.modelpath` path.
      I'd suggest setting a `num_threads` parameter here to the number of threads available in the CPU to significantly boost the inference process.
      Remember to allocate tensors (there is a method for it).
    * in `run_inference` method, invoke the model.
    * in `postprocess_outputs`, implement the method for getting the output tensor (check out `preprocess_input` method for hints on how to do it).

* `[4pt]` Finish the `INT8Model` class:

    * In `optimize_model`, optimize a model to work in full `int8` mode:

        * use `tf.lite.Optimize.DEFAULT` optimization,
        * use `calibration_dataset_generator` as `converter.representative_dataset`,
        * use `tf.int8` as inference input and output type.
        * in general, set following members of the converter object: `optimizations`, `representative_dataset`, `target_spec.supported_ops`, `inference_input_type` and `inference_output_type`
    * Implement `prepare_model`, `run_inference`
    * Implement `preprocess_input` and `postprocess_outputs` methods:

        * remember to quantize the inputs and dequantize the outputs (`scale` and `zero_point` parameters are present in `self.model.get_input_details()[0]['quantization']` field, respectively)!

* `[2pt]` Finish the `ImbalancedINT8Model` class:

    * Implement `optimize_model` method, where the `calibration_dataset_generator` will take all examples for objects with 5 class and use them for calibration:
        
        * Use `self.dataset.dataX` and `self.dataset.dataY` to extract all inputs for a particular class.
        * Remember to use self.dataset.prepare_input_sample method.

* In the main script, uncomment all already supported classes and run it (it may take some time):

  ```
  python3 -m dl_in_iot_course.l02_quantization.quantization_experiments \
        --model-path models/pet-dataset-tensorflow.h5 \
        --dataset-root build/pet-dataset/ \
        --results-path build/results
  ```

  In the `build/results` directory, the script will create:
    
    * `<prefix>-metrics.md` file - contains basic metrics, such as accuracy, precision, sensitivity or G-Mean, along with inference time
    * `<prefix>-confusion-matrix.png` file - contains visualization of confusion matrix for the model evaluation.
  Those files will be created for:

    * `native` - the model running in TensorFlow framework,
    * `tflite-fp32` - the model running in TFLite runtime with FP32 precision,
    * `tflite-int8-<calibsize>` - the model running in TFLite runtime with INT8 precision calibrated with `<calibsize>` fraction of training dataset,
    * `tflite-imbint8` - the model running in TFLite runtime with INT8 precision calibrated with samples for several classes.

  `NOTE:` To download the dataset, add `--download-dataset` flag.

  `NOTE:` If the evaluation takes too long, reduce the test dataset size by setting `--test-dataset-fraction` to some lower value, but inform about this in the Summary note.

* Write a small summary for experiments containing:

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

`NOTE:` the INT8 models may actually perform slower than FP32 models on x86_64 CPUs.

`NOTE:` there is no need to include the models in the repository.

`NOTE:` confusion matrix shows clearly if there are any issues with the optimized model.
If the confusion matrix is almost random (with no significantly higher values along the diagonal) - there are possible issues with the model, usually within preprocessing step (make sure to use `scale`, `zero_point` parameters and to convert the input data to `int8` type).

## Resources

Most of the documentation regarding quantization in TensorFlow Lite can be found in:

* [TFLite Converter documentation](https://www.tensorflow.org/lite/convert)
* [TFLite Interpreter documentation](https://www.tensorflow.org/lite/guide/inference)
* [TFLite Post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)
* [TFLite Quantization specification](https://www.tensorflow.org/lite/performance/quantization_spec)
