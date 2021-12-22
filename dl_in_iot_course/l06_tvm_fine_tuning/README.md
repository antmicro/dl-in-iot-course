# Apache TVM fine-tuning

The aim of this laboratory is to check the work of Apache TVM model fine-tuning feature.

## Tasks

* Go through the [Auto-tuning a Convolutional Network for x86 CPU](https://tvm.apache.org/docs/how_to/tune_with_autotvm/tune_relay_x86.html#sphx-glr-how-to-tune-with-autotvm-tune-relay-x86-py).
* Go [fine_tuning_experiments script](fine_tuning_experiments.py) and check what it does.
* The class `TVMFineTunedModel` inherits following methods from [TVMModel from l04_tvm assignment](../l04_tvm/tvm_experiments.py):

    * `preprocess_input`,
    * `postprocess_outputs`,
    * `prepare_model`,
    * `run_inference`.
* Use the implemented `TVMModel` from L04 assignment or implement above-mentioned methods.
* `[5pt]` Implement `tune_kernels` method:

    * Use `get_tuner` method for each task from tasks,
    * Use `len(task.config_space)` as `n_trial` for `tuner.tune`
    * Use `measure_option` in tuning method,
    * Use `autotvm.callback.progress_bar` and `autotvm.callback.log_to_file` callbacks (use `self.optlogpath` as log path).
    * Add early stopping after 20% of `n_trial` trials with no improvement.
* `[6pt]` Implement `tune_graph` method:

    * Focus on `nn.conv2d` operator only (use `relay.op.get`).
    * Use `PBQPTuner` as tuning executor.
    * Use `mod`, `self.input_name`, `self.input_shape`, `self.optlogpath`, `self.target` variables from `tune_kernels` and `optimize_model` to set up the executor.
    * Use `benchmark_layout_transform` for setting up benchmarks (use `min_exec_num` 5).
    * Run the executor.
    * Save tuning results to `self.graphoptlogpath` file.
* `[4pt]` Finish the implementation of model fine-tuning:

    * Extract tasks from the initially optimized module using `autotvm.task.extract_from_program` (focus on `nn.conv2d` operator only).
    * Tune kernels using `tune_kernels` method.
    * Tune the whole graph using `tune_graph` method.
    * Compile and save the model library - use `autotvm.apply_graph_best` method to load log from `self.graphoptlogpath`.
* Run benchmarks using:

    ```
    python3 -m dl_in_iot_course.l06_tvm_fine_tuning.fine_tuning_experiments \
        --fp32-model-path models/pet-dataset-tensorflow.fp32.tflite \
        --dataset-root build/pet-dataset/ \
        --results-path build/fine-tuning-results
    ```
  `NOTE:` The fine-tuning takes quite a long time.
* `[2pt]` In directory for assignment's summary, include:

    * Kernel log file (`pet-dataset-tensorflow.fp32.tvm-tune.kernellog`),
    * Graph log file (`pet-dataset-tensorflow.fp32.tvm-tune.graphlog`).
* Write a very brief summary:

    * `[1pt]` Compare the inference time between the fine-tuned model and FP32 model with NCHW layout with opt level 3 and `llvm -mcpu=core-avx2` target.

At least a very slight improvement should be observed.

There should be no need for additional imports.
The blocks of code to implement (3 blocks) should take at most around 20 lines.

Additional factors:

* `[2pt]` Git history quality

## Resources

* [Auto-Tune with Templates and AutoTVM](https://tvm.apache.org/docs/how_to/tune_with_autotvm/tune_relay_x86.html)
* [tvm.autotvm](https://tvm.apache.org/docs/reference/api/python/autotvm.html)
* [tvm.autotvm.tuner.Tuner.tune](https://tvm.apache.org/docs/reference/api/python/autotvm.html#tvm.autotvm.tuner.Tuner.tune)
* [tvm.autotvm.graph_tuner.PBQPTuner](https://github.com/apache/tvm/blob/4e0bf23a963d7464abb05e44cf11884d56c05d1c/python/tvm/autotvm/graph_tuner/pbqp_tuner.py#L24)
* [tvm.autotvm.graph_tuner.BaseGraphTuner](https://github.com/apache/tvm/blob/4e0bf23a963d7464abb05e44cf11884d56c05d1c/python/tvm/autotvm/graph_tuner/base_graph_tuner.py#L74)
* [tvm.autotvm.task.extract_from_program](https://github.com/apache/tvm/blob/4e0bf23a963d7464abb05e44cf11884d56c05d1c/python/tvm/autotvm/task/relay_integration.py#L58)
* [tvm.autotvm.apply_graph_best](https://github.com/apache/tvm/blob/4e0bf23a963d7464abb05e44cf11884d56c05d1c/python/tvm/autotvm/task/dispatcher.py#L370)
