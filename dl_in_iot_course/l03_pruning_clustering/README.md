# Optimization Algorithms - Pruning and Clustering

The aim of this laboratory is to demonstrate the pruning and clustering of deep learning models.
For this task, we will use [TensorFlow Model Optimization Toolkit](https://github.com/tensorflow/model-optimization).

## Tasks

### Theoretical task - Quantization follow-up

Demonstrate why the asymmetric quantization is more computationally-demanding than the symmetric quantization.

Start from the fact that in symmetric quantization we have:

    WX = s_W * (W_int) * s_X * (X_int)

And in asymmetric quantization we have:

    WX = s_W * (W_int - z_W) * s_X * (X_int - z_X)

Simplify the above formulas, show the common part of the equations, show the differences, tell what can be optimized and what brings an overhead.

### Practical tasks - Pruning and Clustering

Coming soon

## Resources

* [TensorFlow Model Optimization Pruning documentation](https://www.tensorflow.org/model_optimization/guide/pruning)
* [TensorFlow Model Optimization Clustering documentation](https://www.tensorflow.org/model_optimization/guide/clustering)
