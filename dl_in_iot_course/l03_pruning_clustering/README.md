# Optimization Algorithms - Pruning and Clustering

The aim of this laboratory is to demonstrate the pruning and clustering of deep learning models.
For this task, we will use [TensorFlow Model Optimization Toolkit](https://github.com/tensorflow/model-optimization) and [Neural Network Intelligence (NNI) Toolkit](https://github.com/microsoft/nni).

## Unstructured pruning and clustering with TFMOT

### Tasks

* Go over the [pruning_clustering_experiments script](pruning_clustering_experiments.py) and check what it does (especially check the `optimize_model` method that prepares structures for model optimization and fine-tuning).
* In the end of the afore-mentioned `optimize_model` function there is a `compress_and_fine_tune` method that will need to be completed in the following tasks - it will handle model optimization and fine-tuning
* `[2pt]` Finish the implementation of the `TFMOTOptimizedModel` class:

    * This class is supposed to firstly optimize and fine-tune the Keras model using TensorFlow Optimization Toolkit, and then compile it to TensorFlow Lite in FP32 mode for inference purposes.
    * Implement the `prepare_model`, `preprocess_input`, `run_inference` and `postprocess_outputs` methods.

      `NOTE:` If you completed the `l02_quantization` tasks, just set the parent class of the `TFMOTOptimizedModel` to `FP32Model` from `dl_in_iot_course.l02_quantization.quantization_experiments` - all methods for preparing a model, running inference and processing inputs and outputs will be reused.
* `[4pt]` Finish the implementation of the model clustering in `ClusteredModel` class. In the `compress_and_fine_tune` method:

    * Load the Keras model.
    * Make the entire model trainable - check out `trainable` parameter.
    * Create a clustered model - use a proper `tfmot` method for adding clustering to model, use `self.num_clusters` to set the number of clusters, use linear centroid initialization.
    * Fine-tune the model - compile and fit the model using `self.optimizer`, `self.loss`, `self.metrics`, `self.traindataset`, `self.epochs`, `self.validdataset` objects created in `optimize_model` method.
    * Remove (strip) all clustering-related data from the model
    * Convert the model to FP32 TFLite model and save to `self.modelpath`.
* `[4pt]` Finish the implementation of the model pruning in `PrunedModel` class:

    * The `compress_and_fine_tune` method comes with number of epochs and a simple pruning schedule.
    * Load the Keras model.
    * Create a pruned model - use the `prune_low_magnitude` pruning along with the `sched` schedule.
    * Compile the model using `self.optimizer`, `self.loss`, `self.metrics`, `self.traindataset`, `self.epochs`, `self.validdataset` objects created in `optimize_model` method.
    * Set up `UpdatePruningStep` (and optionally `PruningSummaries`) callbacks.
    * Fit the model.
    * Convert the model to FP32 TFLite model and save to `self.modelpath`.

* The implemented elements of the task can be executed using:

  ```
  python3 -m dl_in_iot_course.l03_pruning_clustering.pruning_clustering_experiments \
        --model-path models/pet-dataset-tensorflow.h5 \
        --dataset-root build/pet-dataset/ \
        --results-path build/results --tasks all
  ```
  Where `--tasks` can be one of:
  * `all` - run all tests
  * `clustered-all` - run all clustering tasks
  * `pruned-all` - run all pruning tasks
  * `pruned-<level>-fp32` - run specific pruning task (check `--help` for variants)
  * `clustered-<num_clusters>-fp32` - run specific clustering task (check `--help` for variants)

* In the `build/results` directory, the script will create:

  * `<prefix>-metrics.md` file - contains basic metrics, such as accuracy, precision, sensitivity or G-Mean, along with inference time
  * `<prefix>-confusion-matrix.png` file - contains visualization of confusion matrix for the model evaluation.
  Those files will be created for:

  * `clustered-<num_clusters>-fp32` - the clustered model with `num_clusters` clusters.
  * `pruned-<sparsity>-fp32` - the pruned model.

  `NOTE:` To download the dataset, add `--download-dataset` flag.
  The dataset downloaded from the previous task can be reused.

  `NOTE:` If the evaluation takes too long, reduce the test dataset size by setting `--test-dataset-fraction` to some lower value, but inform about this in the Summary note.

* Write a small summary for experiments containing:

  * Performance and quality data for each experiment:

    * The computed metrics,
    * Confusion matrix.
  * `[2pt]` Write the deflation percentage along with the size of the compressed models with ZIP tool.
    In the directory containing `.tflite` files, you can run (check [sharkdp/fd](https://github.com/sharkdp/fd) for details):
    ```
    fd -t f -e tflite -x zip {.}.zip {}
    ```
    `NOTE:` in Singularity/Docker environments the `fd` tool is named `fdfind`.
  * Answers for the questions:

    * `[1pt]` In their best variants, how do quantization, pruning and clustering compare to each other (both performance- and quality-wise)?
    * `[1pt]` How do the sizes of compressed models compare to the size of the TFLite FP32 model (how many times is each of the models smaller than the uncompressed FP32 solution).
    * `[1pt]` What does deflation means in ZIP tool and how does it correspond to the observed model optimizations?
    * `[1pt]` Which of the compression methods gives the smallest and best-performing model (is there a dominating solution on both factors)?

    The summary should be put in the project's `summaries` directory - follow the README.md in this directory for details.


`NOTE:` each of the required code snippets should take around 30 lines, but in general less

Additional factors:

* `[2pt]` Git history quality

## Structured pruning with NNI

In this task we will work on a much simpler model trained for [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).
Since NNI supports structured pruning mainly in PyTorch, we will switch to this framework for this task.

### Tasks

* Read the [structured_pruning script](structured_pruning_experiments.py) thoroughly - in the beginning of the script there are several "constants" that need to be used later in the script.
* `NOTE:` You DO NOT NEED to train the model, use the existing model `fashion-mnist-classifier.pth` from `models` directory.
* `[1pt]` Create a traced optimizer (check `nni.trace` method), use Adam optimizer with `TRAINING_LEARNING_RATE` (as in training optimizer).
* `[1pt]` Formulate the configuration list for the ActivationAPoZRank pruner - we want to prune both 2D convolutions and linear layers:
  * Use `SPARSITY` (defined in the beginning of the script) total sparsity
  * Prune `Conv2d` and `Linear` layers
* `[1pt]` What is more, NNI by default prunes ALL layers of given type, even the output ones - EXCLUDE the final linear layer from pruning schedule (check name of the layer and add `exclude` entry with given `op_names`).
* `[1pt]` Define `ActivationAPoZRank` (or other activation-based) pruner using the `model`, defined configuration list, `trainer` method, traced optimizer and criterion. Set `training_batches` to 1000. I highly recommend using legacy version of the pruner (documented in [ActivationAPoZRank v2.8](https://nni.readthedocs.io/en/v2.8/reference/compression/pruner.html#activation-apoz-rank-pruner)
* `[1pt]` The `pruner.compress()` method will compute the pruning mask, and the additional prints will show the pruning status - Please include logs from terminal for those printouts
* `[2pt]` Speedup the model using `ModelSpeedup.speedup_model` function. The expected dummy input for the network should have 1x1x28x28 shape (use `torch.randn(shape).to(model.device)`).
* `[3pt]` Fine-tune the model:

    * Define the optimizer (use Adam with `FINETUNE_LEARNING_RATE`)
    * Train the model using `model.train_model` for `FINE_TUNE_EPOCHS` epochs.
* `[2pt]` Implement `convert_to_onnx` method to save the model to ONNX format (`torch` has methods for exporting to ONNX).
* `[2pt]` Implement `convert_to_tflite` method to convert the ONNX model to FP32 TensorFlow Lite model.
  Use `ai_edge_torch` module, or optionally `onnx_tf` to perform `torch->onnx->tflite` conversion.
  The latter approach won't work for recent releases of TensorFlow.
  It is also possible to try out `onnx2tf` - the provided `requirements.txt` and Docker image do not provide dependencies for this, but it is possible to try out.
* `[1pt]` Use [Netron](https://github.com/lutzroeder/netron) tool to visualize the network
* In summary:
    * `[1pt]` Include the shape of the model before pruning (script logs provide those, as well as other data), along with its accuracy and inference time
    * `[1pt]` Include the pruning logs collected around `pruning.compress()`
    * `[1pt]` Include the shape of the model after pruning, along with its accuracy and inference time before fine-tuning
    * `[1pt]` Include the accuracy and inference time of the model after fine-tuning
    * `[1pt]` Include the fine-tuned PyTorch model in the summary directory (use Git LFS, check `git lfs install`, `git lfs track` commands, you need to apply them before adding the file and committing it)
    * `[1pt]` Include the ONNX file with pruned PyTorch model (use Git LFS)
    * `[1pt]` Include the TFLite file with pruned PyTorch model (use Git LFS)
    * `[1pt]` In the summary, include the visualization of the graph using Netron

The command should be executed more or less as follows:

```
python3 -m dl_in_iot_course.l03_pruning_clustering.structured_pruning_experiments \
    --input-model models/fashion-mnist-classifier.pth \
    --backup-model backup-model.pth \
    --final-model final-model.pth \
    --dataset-path fashion-dataset \
    --onnx-model model.onnx \
    --tflite-model model.tflite
```

Additional factors:

* `[2pt]` Git history quality

## Resources

* [TensorFlow Model Optimization Pruning documentation](https://www.tensorflow.org/model_optimization/guide/pruning)
* [TensorFlow Model Optimization Clustering documentation](https://www.tensorflow.org/model_optimization/guide/clustering)
* [NNI documentation](https://nni.readthedocs.io/en/v2.8/)
* [NNI Model Compression documentation](https://nni.readthedocs.io/en/v2.8/compression/overview.html)
* [NNI Pruning quickstart](https://nni.readthedocs.io/en/v2.8/tutorials/pruning_quick_start_mnist.html)
* [Activation APoZ Rank Pruner doc](https://nni.readthedocs.io/en/v2.8/reference/compression/pruner.html#activation-apoz-rank-pruner)
