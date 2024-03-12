import time
from pathlib import Path
from typing import Optional, Any
from tqdm import tqdm
import numpy as np

from dl_in_iot_course.misc.pet_dataset import PetDataset
from dl_in_iot_course.misc.draw import draw_confusion_matrix
from dl_in_iot_course.misc import metrics


class ModelTester(object):
    """
    This is an abstract class for running evaluation on given models.
    """

    def __init__(
        self,
        dataset: PetDataset,
        modelpath: Path,
        originalmodel: Optional[Path] = None,
        logdir: Optional[Path] = None,
    ):
        """
        Initializer for ModelTester.

        Parameters
        ----------
        dataset : PetDataset
            A dataset object to test on
        modelpath : Path
            Path to the model to test
        originalmodel : Path
            Path to the model to optimize before testing.
            Optimized model will be saved in modelpath
        logdir : Path
            Path to the training/optimization logs
        """
        self.dataset = dataset
        self.modelpath = modelpath
        self.logdir = logdir
        if self.logdir:
            self.logdir.mkdir(parents=True, exist_ok=True)
        self.timemeasurements = []
        if originalmodel:
            self.optimize_model(originalmodel)
        self.prepare_model()

    def reset_measurements(self):
        """
        Resets the dataset and model measurements.
        """
        self.timemeasurements = []
        self.dataset.reset_metrics()

    def _run_inference(self) -> Optional[Any]:
        """
        Wraps the inference passes with time measurements.

        Returns
        -------
        Optional[Any] : the method passes the results from run_inference
        """
        start = time.perf_counter()
        result = self.run_inference()
        duration = time.perf_counter() - start
        # measure time in milliseconds
        self.timemeasurements.append(duration * 1000.0)
        return result

    def test_inference(
        self, resultspath: Path, prefix: str, testdatasetpercentage: float = 0.3
    ):
        """
        Runs inference on test data and evaluates the model.

        The inference performance results are saved in resultspath with prefix.

        Parameters
        ----------
        resultspath : Path
            The path where results will be stored
        prefix : str
            The prefix for the performance files
        testdatasetpercentage : float
            The percentage of the test dataset to use for the evaluation
        """
        self.reset_measurements()
        if testdatasetpercentage == 1.0:
            dataX = self.dataset.testX
            dataY = self.dataset.testY
        else:
            _, dataX, _, dataY = self.dataset.split_dataset(
                percentage=testdatasetpercentage, usetest=True
            )
        # for each entry in the test dataset
        for X, y in tqdm(list(zip(dataX, dataY)), desc=f"evaluating {prefix}"):
            # preprocess data
            Xp = self.dataset.prepare_input_sample(X)
            yp = self.dataset.prepare_output_sample(y)
            self.preprocess_input(Xp)
            # run the inference
            preds = self._run_inference()
            posty = self.postprocess_outputs(preds)
            # evaluate the current data
            self.dataset.evaluate(posty, yp)
        # draw the final confusion matrix
        draw_confusion_matrix(
            self.dataset.confusion_matrix,
            resultspath / f"{prefix}-confusion-matrix.png",
            "Confusion matrix",
            self.dataset.classnames,
        )
        # generate a file
        with open(resultspath / f"{prefix}-metrics.md", "w") as metfile:
            conf_matrix = self.dataset.confusion_matrix
            metfile.writelines(
                [
                    f"Model type: {prefix}\n\n",
                    f"* Accuracy: {metrics.accuracy(conf_matrix)}\n",
                    f"* Mean precision: {metrics.mean_precision(conf_matrix)}\n",
                    f"* Mean sensitivity: {metrics.mean_sensitivity(conf_matrix)}\n",  # noqa: E501
                    f"* G-Mean: {metrics.g_mean(conf_matrix)}\n",
                    f"* Mean inference time: {np.mean(self.timemeasurements[1:])} ms\n"  # noqa: E501
                    f"* Top-5 percentage: {self.dataset.top_5_count / self.dataset.total}\n",  # noqa: E501
                ]
            )

    def preprocess_input(self, X: Any):
        """
        Preprocesses the inputs so they are usable in the model.

        This includes both data preprocessing, and proper data allocation.

        The data should be stored in appropriate location in this method so it
        can be used later in the run_inference method (i.e. stored as class
        field or added to allocated space of the model in TensorFlow Lite).

        Parameters
        ----------
        X : Any
            Internal representation of the input that is subjected to
            preprocessing.
        """
        raise NotImplementedError

    def postprocess_outputs(self, Y: Optional[Any]) -> np.ndarray:
        """
        Postprocesses the outputs and return them in a form of NumPy array.

        This method takes the output of the model either from the parameter Y,
        which is the output of the run_inference method, or from the internal
        structures of the model if the prediction method does not support
        returning values directly (i.e. in TensorFlow Lite models).

        The method also postprocesses the outputs so they are returned in a
        form of NumPy array that can be used for the Dataset evaluate method.

        Parameters
        ----------
        Y : Optional[Any]
            The output of the run_inference method

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

    def run_inference(self):
        """
        Runs inference on a given sample and returns predictions.
        The inputs for the models in compiler runtimes usually need to be
        stored in model's allocated structures.

        The inputs need to be properly allocated and ready for inference
        earlier in the preprocess_input method

        Returns
        -------
        Optional[Any] : predictions from the model, if returned by the
            prediction method, otherwise None
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
