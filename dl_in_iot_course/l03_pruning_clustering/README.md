# Optimization Algorithms - Pruning and Clustering

The aim of this laboratory is to demonstrate the pruning and clustering of deep learning models.
For this task, we will use [TensorFlow Model Optimization Toolkit](https://github.com/tensorflow/model-optimization).

## Tasks

* Go over the [pruning_clustering_experiments script](pruning_clustering_experiments.py) and check what it does (especially check the `optimize_model` method that prepares structures for model optimization and fine-tuning).
* In the end of the afore-mentioned `optimize_model` function there is a `compress_and_fine_tune` method that will need to be completed in the following tasks - it will handle model optimization and fine-tuning
* `[2pt]` Finish the implementation of the `TFMOTOptimizedModel` class:

    * This class is supposed to firstly optimize and fine-tune the Keras model using TensorFlow Optimization Toolkit, and then compile it to TensorFlow Lite in FP32 mode for inference purposes.
    * Implement the `prepare_model`, `preprocess_input`, `run_inference` and `postprocess_outputs` methods.

      `NOTE:` If you completed the `l02_quantization` tasks, just set the parent class of the `TFMOTOptimizedModel` to `FP32Model` from `dl_in_iot_course.l02_quantization.quantization_experiments` - all methods for preparing a model, running inference and processing inputs and outputs will be reused.
* `[4pt]` Finish the implementation of the model clustering in `ClusteredModel` class:

    * Load the Keras model.
    * Create a clustered model - use a proper `tfmot` method for adding clustering to model, use `self.num_clusters` to set the number of clusters, use linear centroid initialization.
    * Fine-tune the model - compile and fit the model using `self.optimizer`, `self.loss`, `self.metrics`, `self.traindataset`, `self.epochs`, `self.validdataset` objects created in `optimize_model` method.
    * Convert the model to FP32 TFLite model and save to `self.modelpath`.
* `[4pt]` Finish the implementation of the model clustering in `PrunedModel` class:

    * The `compress_and_fine_tune` method comes with number of epochs and a simple pruning schedule.
    * Load the Keras model.
    * Create a pruned model - use the `prune_low_magnitude` pruning along with the `sched` schedule.
    * Fine-tune the model - compile and fit the model using `self.optimizer`, `self.loss`, `self.metrics`, `self.traindataset`, `self.epochs`, `self.validdataset` objects created in `optimize_model` method.
    * Remember to set up proper callbacks for pruning.

* In the main script, uncomment all already supported classes and run it (it will definitely take some time):

  ```
  python3 -m dl_in_iot_course.l03_pruning_clustering.pruning_clustering_experiments \
        --model-path models/pet-dataset-tensorflow.h5 \
        --dataset-root build/pet-dataset/ \
        --results-path build/results
  ```

  In the `build/results` directory, the script will create:

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

          fd -t f -e tflite -x zip {.}.zip {}

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

## Resources

* [TensorFlow Model Optimization Pruning documentation](https://www.tensorflow.org/model_optimization/guide/pruning)
* [TensorFlow Model Optimization Clustering documentation](https://www.tensorflow.org/model_optimization/guide/clustering)
