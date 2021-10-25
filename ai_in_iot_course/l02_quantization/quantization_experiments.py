import argparse
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import time
import numpy as np

from ai_in_iot_course.misc.pet_dataset import PetDataset
from ai_in_iot_course.misc.draw import draw_confusion_matrix
from ai_in_iot_course.misc import metrics


class ModelTester(object):
    """
    This is an abstract class for running evaluation on given models.
    """

    def __init__(self, dataset: PetDataset, modelpath: Path):
        """
        Initializer for ModelTester.

        Parameters
        ----------
        dataset : a dataset object, i.
        """
        self.dataset = dataset
        self.modelpath = modelpath
        self.timemeasurements = []

    def reset_measurements(self):
        """
        Resets the dataset and model measurements.
        """
        self.timemeasurements = []
        self.dataset.reset_metrics()

    def _run_inference(self, X):
        """
        Wraps the inference passes with time measurements.

        Parameters
        ----------
        X : Single input sample

        Returns
        -------
        Any : predictions from the model
        """
        start = time.perf_counter()
        result = self.run_inference(X)
        duration = time.perf_counter() - start
        # measure time in milliseconds
        self.timemeasurements.append(duration * 1000.0)
        return result

    def test_inference(self, resultspath : Path, prefix : str):
        """
        Runs inference on test data and evaluates the model.

        The inference performance results are saved in resultspath with prefix.

        Parameters
        ----------
        resultspath : Path
            The path where results will be stored
        prefix : str
            The prefix for the performance files
        """
        # for each entry in the test dataset
        for X, y in tqdm(list(zip(self.dataset.testX, self.dataset.testY))):
            # preprocess data
            Xp = self.dataset.prepare_input_sample(X)
            yp = self.dataset.prepare_output_sample(y)
            prepX = self.preprocess_input(Xp)
            # run the inference
            preds = self._run_inference(prepX)
            posty = self.postprocess_outputs(preds)
            # evaluate the current data
            self.dataset.evaluate(posty, yp)
        # draw the final confusion matrix
        draw_confusion_matrix(
            self.dataset.confusion_matrix,
            resultspath / f'{prefix}-confusion-matrix.png',
            'Confusion matrix',
            self.dataset.classnames
        )
        # generate a file
        with open(resultspath / f'{prefix}-metrics.md', 'w') as metfile:
            conf_matrix = self.dataset.confusion_matrix
            metfile.writelines([
                f'Model type: {prefix}\n\n',
                f'* Accuracy: {metrics.accuracy(conf_matrix)}\n',
                f'* Mean precision: {metrics.mean_precision(conf_matrix)}\n',
                f'* Mean sensitivity: {metrics.mean_sensitivity(conf_matrix)}\n',  # noqa: E501
                f'* G-Mean: {metrics.g_mean(conf_matrix)}\n',
                f'* Mean inference time: {np.mean(self.timemeasurements[1:])} ms\n'  # noqa: E501
                f'* Top-5 percentage: {self.dataset.top_5_count / self.dataset.total}\n'  # noqa: E501
            ])

    def preprocess_input(self, X):
        """
        Preprocesses the inputs so they are usable in the model.

        Parameters
        ----------
        X : Inputs from the dataset

        Returns
        -------
        Any : inputs ready for model processing
        """
        return X

    def postprocess_outputs(self, Y):
        """
        Postprocesses the outputs and return them in a form of NumPy array.

        Parameters
        ----------
        Y : outputs from the model

        Returns
        -------
        np.ndarray : Postprocessed output
        """
        return Y

    def prepare_model(self):
        """
        Loads and prepares the model for inference.
        """
        raise NotImplementedError

    def run_inference(self, X):
        """
        Runs inference on a given sample and returns predictions.

        Parameters
        ----------
        X : Single input sample

        Returns
        -------
        Any : predictions from the model
        """
        raise NotImplementedError

    def optimize_model(self, originalmodel: Path):
        """
        Preprocesses a given model and saves it to self.modelpath.

        Parameters
        ----------
        originalmodel : Path
            Path to the model to optimize
        """
        pass


class NativeModel(ModelTester):
    """
    Tests the performance of the model ran natively with TensorFlow.

    This tester verifies the work of the native TensorFlow model without any
    optimizations.
    """
    def prepare_model(self):
        self.model = tf.keras.models.load_model(str(self.modelpath))

    def run_inference(self, X):
        return self.model.predict(X)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-path',
        help='Path to the model file',
        type=Path
    )
    parser.add_argument(
        '--dataset-root',
        help='Path to the dataset file',
        type=Path
    )
    parser.add_argument(
        '--download-dataset',
        help='Download the dataset before training',
        action='store_true'
    )
    parser.add_argument(
        '--results-path',
        help='Path to the results',
        type=Path
    )

    args = parser.parse_args()

    args.results_path.mkdir(parents=True, exist_ok=True)

    dataset = PetDataset(args.dataset_root, args.download_dataset)

    # test of the model executed with TensorFlow
    tester = NativeModel(dataset, args.model_path)
    tester.prepare_model()
    tester.test_inference(args.results_path, 'native')
