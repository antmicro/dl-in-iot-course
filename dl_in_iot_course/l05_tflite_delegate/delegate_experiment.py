import argparse
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import numpy as np
from numpy.testing import assert_almost_equal
import onnx  # noqa: F401
from onnx_tf.backend import prepare  # noqa: F401


def convert_onnx_to_tensorflow(onnxpath: Path, tfpath: Path):
    """
    Converts ONNX model to the TensorFlow model.

    Parameters
    ----------
    onnxpath : Path
        Path to the ONNX model file
    tfpath : Path
        Path to the output TensorFlow model
    """
    # TODO implement
    pass


def convert_to_tflite(tfpath: Path, tflitepath: Path):
    """
    Converts the TensorFlow model from file to the TensorFlow Lite model.

    Parameters
    ----------
    tfpath : Path
        Path to the TensorFlow model
    tflitepath : Path
        Path to the output TensorFlow Lite model
    """
    # TODO implement
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-onnx-model-path",
        help="Path to the ONNX model file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--tensorflow-model-path",
        help="Path to the compiled model file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--compiled-model-path",
        help="Path to the compiled model file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--delegate-path", help="Path to the model delegate", type=Path, required=True
    )
    parser.add_argument(
        "--num-tests", help="Number of tests to conduct", type=int, default=1000
    )

    args = parser.parse_args()

    convert_onnx_to_tensorflow(args.input_onnx_model_path, args.tensorflow_model_path)

    convert_to_tflite(args.tensorflow_model_path, args.compiled_model_path)

    nodelegate = tf.lite.Interpreter(str(args.compiled_model_path))
    nodelegate.allocate_tensors()

    delegate = tf.lite.Interpreter(
        str(args.compiled_model_path)
        # TODO add loading delegate
    )
    delegate.allocate_tensors()

    for _ in tqdm(range(args.num_tests)):
        x = np.random.randint(-200, 200, size=(1, 4)).astype("float32")
        y = np.random.randint(-200, 200, size=(1, 3)).astype("float32")
        nodelegate.set_tensor(nodelegate.get_input_details()[0]["index"], x)
        nodelegate.set_tensor(nodelegate.get_input_details()[1]["index"], y)
        nodelegate.invoke()
        nodelegateres = nodelegate.get_tensor(
            nodelegate.get_output_details()[0]["index"]
        )
        delegate.set_tensor(delegate.get_input_details()[0]["index"], x)
        delegate.set_tensor(delegate.get_input_details()[1]["index"], y)
        delegate.invoke()
        delegateres = delegate.get_tensor(delegate.get_output_details()[0]["index"])
        assert_almost_equal(nodelegateres, delegateres, 0.01)
