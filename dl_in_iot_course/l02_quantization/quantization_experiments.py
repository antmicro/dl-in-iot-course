import argparse
import tensorflow as tf
from pathlib import Path
from typing import Optional

from dl_in_iot_course.misc.pet_dataset import PetDataset
from dl_in_iot_course.misc.modeltester import ModelTester


class NativeModel(ModelTester):
    """
    Tests the performance of the model ran natively with TensorFlow.

    This tester verifies the work of the native TensorFlow model without any
    optimizations.
    """

    def prepare_model(self):
        self.model = tf.keras.models.load_model(str(self.modelpath))
        # TODO print model summary

    def preprocess_input(self, X):
        self.X = X

    def run_inference(self):
        return self.model.predict(self.X, verbose=0)


class FP32Model(ModelTester):
    """
    This tester tests the performance of FP32 TensorFlow Lite model.
    """

    # TODO def optimize_model(self, originalmodel: Path):

    # TODO def prepare_model(self):

    def preprocess_input(self, X):
        # since we only want to measure inference time, not tensor allocation,
        # we mode setting tensor to preprocess_input
        self.model.set_tensor(self.model.get_input_details()[0]["index"], X)

    # TODO def run_inference(self):

    # TODO def postprocess_outputs(self, Y):


class INT8Model(ModelTester):
    """
    This tester tests the performance of FP32 TensorFlow Lite model.
    """

    def __init__(
        self,
        dataset: PetDataset,
        modelpath: Path,
        originalmodel: Optional[Path] = None,
        calibrationdatasetpercent: float = 0.5,
    ):
        """
        Initializer for INT8Model.

        Parameters
        ----------
        dataset : PetDataset
            A dataset object to test on
        modelpath : Path
            Path to the model to test
        originalmodel : Path
            Path to the model to optimize before testing.
            Optimized model will be saved in modelpath
        calibrationdatasetpercent : float
            Tells the percentage of train dataset used for calibration process
        """

        self.calibrationdatasetpercent = calibrationdatasetpercent

        super().__init__(dataset, modelpath, originalmodel)

    def optimize_model(self, originalmodel: Path):
        def calibration_dataset_generator():
            return self.dataset.calibration_dataset_generator(
                self.calibrationdatasetpercent, 1234
            )

        # TODO finish implementation

    # TODO def prepare_model(self):

    # TODO def preprocess_input(self, X):

    # TODO def run_inference(self):

    # TODO def postprocess_outputs(self, Y):


class ImbalancedINT8Model(INT8Model):
    def optimize_model(self, originalmodel: Path):
        # TODO implement
        pass


if __name__ == "__main__":
    tasks = {}

    def native(dataset: PetDataset, args: argparse.Namespace):
        # test of the model executed natively
        tester = NativeModel(dataset, args.model_path)
        tester.test_inference(args.results_path, "native", args.test_dataset_fraction)

    tasks["native"] = native

    def tflite_fp32(dataset: PetDataset, args: argparse.Namespace):
        # test of the model executed with FP32 precision
        tester = FP32Model(
            dataset,
            args.results_path / f"{args.model_path.stem}.fp32.tflite",
            args.model_path,
        )
        tester.test_inference(
            args.results_path, "tflite-fp32", args.test_dataset_fraction
        )

    tasks["tflite-fp32"] = tflite_fp32

    def tflite_int8_X(dataset: PetDataset, args: argparse.Namespace, calibsize: float):
        # test of the model executed with INT8 precision
        tester = INT8Model(
            dataset,
            args.results_path / f"{args.model_path.stem}.int8-{calibsize}.tflite",  # noqa: E501
            args.model_path,
            calibsize,
        )
        tester.test_inference(
            args.results_path, f"tflite-int8-{calibsize}", args.test_dataset_fraction
        )

    for calibration_size in [0.01, 0.08, 0.3, 0.8]:
        tasks[f"tflite-int8-{calibration_size}"] = (
            lambda dataset, args, calib_size=calibration_size: tflite_int8_X(
                dataset, args, calib_size
            )
        )

    def tflite_imbint8(dataset: PetDataset, args: argparse.Namespace):
        # test of the model executed with imbalanced INT8 precision
        tester = ImbalancedINT8Model(
            dataset,
            args.results_path / f"{args.model_path.stem}.imbint8.tflite",
            args.model_path,
        )
        tester.test_inference(
            args.results_path, "tflite-imbint8", args.test_dataset_fraction
        )

    tasks["tflite-imbint8"] = tflite_imbint8

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", help="Path to the model file", type=Path)
    parser.add_argument("--dataset-root", help="Path to the dataset file", type=Path)
    parser.add_argument(
        "--download-dataset",
        help="Download the dataset before training",
        action="store_true",
    )
    parser.add_argument("--results-path", help="Path to the results", type=Path)
    parser.add_argument(
        "--test-dataset-fraction",
        help="What fraction of the test dataset should be used for evaluation",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--tasks",
        choices=list(tasks.keys()) + ["all", "tflite-int8-all"],
        nargs="+",
        default=["native"],
    )

    args = parser.parse_args()

    chosen_tasks = []

    for task in args.tasks:
        if task == "all":
            chosen_tasks.extend([t for t in tasks.keys() if t not in chosen_tasks])
        elif task == "tflite-int8-all":
            chosen_tasks.extend(
                [
                    t
                    for t in tasks.keys()
                    if t not in chosen_tasks and t.startswith("tflite-int8")
                ]
            )
        elif task not in chosen_tasks:
            chosen_tasks.append(task)

    args.results_path.mkdir(parents=True, exist_ok=True)

    dataset = PetDataset(args.dataset_root, args.download_dataset)

    for variant in chosen_tasks:
        tasks[variant](dataset, args)
