# Apache TVM DNN compiler

The aim of this laboratory is to show the work of the [Apache TVM framework](https://tvm.apache.org/).

## Tasks

* Go over the [tvm_experiments script](tvm_experiments.py) and check what it does.
* The class `TVMModel` performs the compilation of the model.
  The base of the class is `ModelTester`.
  There are new arguments introduced:

    * `target` - the device to run the model on,
    * `target_host` - the CPU variant communicating with the target,
    * `opt_level` - optimization level for models,
    * `use_nchw_layout` - by default, the TFLite models are accepting images in NHWC format, this flag orders the TVM to flip it.
* `[3pt]` Implement the `preprocess_input` and `postprocess_outputs` methods:

    * Instead of using input names, the Module's `set_input` method accepts input IDs, so you can write i.e. `self.model.set_input(0, tvm.nd.array(X))`.
      Similarly with the outputs
    * Based on `self.quantized` flag apply or not apply quantization and dequantization of inputs and outputs, respectively.

* `[2pt]` Implement the `prepare_model` method:

    * Load the model using TVM runtime
    * Use `tvm.runtime.load_module` to load the module
    * get `default` function from the module (`get_function`)
    * create the context for the used device (use `tvm.runtime.device(self.target)` - it accepts target strings as inputs and returns proper device)
    * using `tvm.contrib.graph_executor.GraphModule` create the model

* `[1pt]` Implement the `run_inference` method
* `[4pt]` Finish the implementation of the model optimization:

    * Load model from TFLite using the `tflite_model` variable declared in the function,
    * Create a `transform.PassContext` (using `with`) with `self.opt_level` opt level,
    * Within the context, apply the transformations wrapped in `seq` variable on `mod` IRModule returned by Relay frontend,
    * Build the library using `relay.build`, `mod` and `params` variables returned by TFLite frontend, `self.target`, and `self.target_host`.
    * Export the library to `self.modelpath`

* Uncomment and run all the experiments in the main script using:

  ```
  python3 -m dl_in_iot_course.l04_tvm.tvm_experiments \
         --fp32-model-path models/pet-dataset-tensorflow.fp32.tflite \
         --int8-model-path models/pet-dataset-tensorflow.int8-0.8.tflite \
         --dataset-root build/pet-dataset/ \
         --target "llvm" \
         --results-path build/results
  ```

* While running:

    * For `TVM MODEL WITH NHWC LAYOUT` check out the TVM log - what is said about the `conv2d`?
    * Compare the log from `TVM MODEL WITH NHWC LAYOUT` with the one in the `TVM MODEL WITH NCHW LAYOUT ...`.
      Also, compare the inference time of both solutions.
    * Check out the log for `TVM PRE-QUANTIZED MODEL WITH NCHW LAYOUT` - it should report similar issues present for the NHWC layout.

* `[2pt]` Based on the log for `TVM PRE-QUANTIZED MODEL WITH NCHW LAYOUT`, update the `relay.transform.ConvertLayout` transform so it supports the missing op for conversion - after adding proper op there should be no issues reported in the log.
* After fixing the transform, run the experiments (at least the pre-quantized experiment) again to collect the final results.
* Assuming the target CPU has Advanced Vector Extensions v2 (AVX2) instruction set (it can be verified in Linux using `lscpu` command), run the script again for all experiments but using the following target:

  ```
  llvm -mcpu=core-avx2
  ```
  In case there is absolutely no performance boost (or if an error occurs during compilation or runtime), please point it out in the summary and enclose the full CPU model name.
* Write a summary for experiments containing:

    * Performance and quality data for each experiment:

        * The computed metrics,
        * Confusion matrix.
    * `[1pt]` Include unique lines of logs for compiled models - NHWC model, INT8 model before and after fixing the conversion to NCHW.
      Extract manually the lines that describe the sources of performance decreases.
    * Include the performance and quality data for the used FP32 TFLite and INT8 TFLite models for comparison purposes.
    * `[1pt]` Compare the sizes of saved compiled models for TFLite and TVM.
    * Answer the questions:

        * `[1pt]` How does the TFLite FP32 model compare to the NHWC-compiled TVM model?
        * `[1pt]` How does the TFLite FP32 model compare to the fastest NCHW-compiled TVM model?
        * `[1pt]` How did switching from NHWC to NCHW in FP32 and INT8 models affect the performance of the model (compare the NHWC and NCHW models with opt level 3)?
        * `[1pt]` How does the TFLite INT8 model compare to the INT8 TVM model (after fixing layout conversion)?
        * `[1pt]` How does the models with opt levels 1, 2, 3, 4 compare to each other performance- and quality-wise?
        * `[1pt]` How using the AVX2 instruction set affected all of the compiled models?

  The summary should be put in the project's `summaries` directory - follow the README.md in this directory for details.


`NOTE:` each of the required code snippets should take around 15 lines, but in general much less

Additional factors:

* `[2pt]` Git history quality

## Resources

* [tvm.target](https://tvm.apache.org/docs/reference/api/python/target.html)
* [TVM tutorials for compiling DNN models](https://tvm.apache.org/docs/how_to/compile_models/index.html)
* [TVM Python API](https://tvm.apache.org/docs/reference/api/python/index.html)
* [tvm.relay.frontend](https://tvm.apache.org/docs/reference/api/python/relay/frontend.html)
* [tvm.relay.transform](https://tvm.apache.org/docs/reference/api/python/relay/transform.html)
* [tvm.contrib.graph_executor](https://tvm.apache.org/docs/reference/api/python/graph_executor.html)
* [tvm.contrib.graph_executor.GraphModule](https://tvm.apache.org/docs/reference/api/python/graph_executor.html?highlight=get_output#tvm.contrib.graph_executor.GraphModule)
