# TensorFlow Lite delegates

The aim of this laboratory is to enhance a simple TensorFlow Lite delegate.

## Tasks

* Check the [delegate-example](../../delegate-example) subdirectory - go through the CMakeLists.txt and sources.
* Build the delegate example based on delegate's README.md.
* Check the [delegate_experiment script](delegate_experiment.py).
* `[1pt]` Implement `convert_onnx_to_tensorflow` method - it should use `onnx` module and `prepare` method from `onnx_tf.backend` submodule.
* `[1pt]` Implement `convert_to_tflite` method - it should load the model saved to file in `convert_onnx_to_tensorflow`.
* `[1pt]` Add loading delegate in the delegate_experiment - use `--delegate-path` argument.
* Run the test script using `models/test-delegate-two-inputs.onnx` model, i.e.:

```
python3 -m dl_in_iot_course.l05_tflite_delegate.delegate_experiment \
    --input-onnx-model-path models/test-delegate-two-inputs.onnx \
    --tensorflow-model-path build/test-delegate-model.pb \
    --compiled-model-path build/test-delegate.tflite \
    --delegate-path ./delegate-example/build/libeigen-delegate.so
```

* `[1pt]` Add Eigen-based implementation of addition in the delegate_experiment - use `Eigen::Map` and `Eigen::VectorXf` classes to implement it (remember to rebuild the C++ project after modifications).
* `[1pt]` Make sure the delegate_experiment script runs properly and doesn't fail using the [delegate_experiment script](delegate_experiment.py) (there should be no assert failure or any other error).

Additional factors:

* `[1pt]` Git history quality

## Resources

* [Implementing a Custom Delegate (TensorFlow Lite documentation)](https://www.tensorflow.org/lite/performance/implementing_delegate)
* [Eigen documentation](https://eigen.tuxfamily.org/dox/)
