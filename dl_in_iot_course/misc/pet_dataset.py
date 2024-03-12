"""
The Oxford-IIIT Pet Dataset wrapper.
"""

import tempfile
from pathlib import Path
import tarfile
from PIL import Image
import numpy as np
from typing import Tuple, List, Any
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dl_in_iot_course.misc.utils import download_url


class PetDataset(object):
    """
    The Oxford-IIIT Pet Dataset

    Omkar M Parkhi and Andrea Vedaldi and Andrew Zisserman and C. V. Jawahar

    It is a classification dataset with 37 classes, where 12 classes represent
    cat breeds, and the remaining 25 classes represent dog breeds.

    It is a seemingly balanced dataset breed-wise, with around 200 images
    examples per class.

    There are 7349 images in total, where 2371 images are cat images, and the
    4978 images are dog images.

    *License*: Creative Commons Attribution-ShareAlike 4.0 International
    License.

    *Page*: `Pet Dataset site <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_.

    The images can be either classified by species (2 classes)
    or breeds (37 classes).

    The affinity of images to classes is taken from annotations, but the class
    IDs are starting from 0 instead of 1, as in the annotations.
    """

    def __init__(self, root: Path, download_dataset: bool = False):
        """
        Prepares all structures and data required for providing data samples.

        Parameters
        ----------
        root : Path
            The path to the dataset data
        download_dataset : bool
            True if dataset should be downloaded first
        """
        self.root = Path(root)

        self.numclasses = None
        self.classnames = dict()

        self.dataX = []
        self.dataY = []

        self.testX = []
        self.testY = []

        if download_dataset:
            self.download_dataset()
        self.prepare()

    def prepare(self):
        """
        Prepares dataX, dataY, testX and testY attributes based on dataset.

        Those lists will store file paths and classes for objects.
        """
        with open(self.root / "annotations" / "trainval.txt", "r") as datadesc:
            for line in datadesc:
                if line.startswith("#"):
                    continue
                fields = line.split(" ")
                self.dataX.append(str(self.root / "images" / (fields[0] + ".jpg")))
                self.dataY.append(int(fields[1]) - 1)
                clsname = fields[0].rsplit("_", 1)[0]
                if self.dataY[-1] not in self.classnames:
                    self.classnames[self.dataY[-1]] = clsname
                assert self.classnames[self.dataY[-1]] == clsname
            self.numclasses = len(self.classnames)
        with open(self.root / "annotations" / "test.txt", "r") as datadesc:
            for line in datadesc:
                if line.startswith("#"):
                    continue
                fields = line.split(" ")
                self.testX.append(str(self.root / "images" / (fields[0] + ".jpg")))
                self.testY.append(int(fields[1]) - 1)
        self.reset_metrics()
        self.mean, self.std = self.get_input_mean_std()
        self.onehotvectors = np.eye(self.numclasses)

    def reset_metrics(self):
        """
        Resets metrics set for evaluation purposes.
        """
        self.confusion_matrix = np.zeros((self.numclasses, self.numclasses))
        self.top_5_count = 0
        self.total = 0

    def download_dataset(self):
        """
        Downloads the dataset to the root directory defined in the constructor.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        imgs = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
        anns = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"  # noqa: E501
        with tempfile.TemporaryDirectory() as tmpdir:
            tarimgspath = Path(tmpdir) / "dataset.tar.gz"
            tarannspath = Path(tmpdir) / "annotations.tar.gz"
            download_url(imgs, tarimgspath)
            download_url(anns, tarannspath)
            tf = tarfile.open(tarimgspath)
            tf.extractall(self.root)
            tf = tarfile.open(tarannspath)
            tf.extractall(self.root)

    def get_data(self) -> Tuple[List, List]:
        """
        Returns the tuple of all inputs and outputs for the dataset.
        """
        return (self.dataX, self.dataY)

    def split_dataset(
        self, percentage: float = 0.25, seed: int = 12345, usetest: bool = False
    ):
        """
        Extracts validation dataset from the train dataset.

        Parameters
        ----------
        percentage : float
            how much data should be taken as validation dataset
        seed : int
            The seed for random state
        usetest : bool
            Tells if the test dataset should be split

        Returns
        -------
        Tuple[List, List, List, List] : Tuple with train inputs, validation
            inputs, train outputs, validation outputs
        """
        dataxtrain, dataxvalid, dataytrain, datayvalid = train_test_split(
            self.testX if usetest else self.dataX,
            self.testY if usetest else self.dataY,
            test_size=percentage,
            random_state=seed,
            shuffle=True,
            stratify=self.testY if usetest else self.dataY,
        )
        return (dataxtrain, dataxvalid, dataytrain, datayvalid)

    def calibration_dataset_generator(
        self, percentage: float = 0.25, seed: int = 12345
    ):
        """
        Creates generator for the calibration data.

        Parameters
        ----------
        percentage : float
            The fraction of data to use for calibration
        seed : int
            The seed for random state
        """
        _, X, _, _ = self.split_dataset(percentage, seed)
        for x in tqdm(X, desc="calibration"):
            yield [self.prepare_input_sample(x)]

    def evaluate(self, predictions: List, truth: List):
        """
        Evaluates the model based on the predictions.

        The method provides various quality metrics determining how well the
        model performs.

        Parameters
        ----------
        predictions : List
            The list of predictions from the model
        truth : List
            The ground truth for given batch
        """
        confusion_matrix = np.zeros((self.numclasses, self.numclasses))
        top_5_count = 0
        confusion_matrix[np.argmax(truth), np.argmax(predictions)] += 1
        top_5_count += 1 if np.argmax(truth) in np.argsort(predictions)[::-1][:5] else 0  # noqa: E501
        self.confusion_matrix += confusion_matrix
        self.top_5_count += top_5_count
        self.total += len(predictions)

    def get_input_mean_std(self) -> Tuple[Any, Any]:
        """
        Returns mean and std values for input tensors.
        Those are precomputed mean and std values for models trained on
        ImageNet dataset.

        Returns
        -------
        Tuple[Any, Any] :
            the standardization values for a given train dataset.
            Tuple of two variables describing mean and std values.
        """
        return np.array([0.485, 0.456, 0.406], dtype="float32"), np.array(
            [0.229, 0.224, 0.225], dtype="float32"
        )  # noqa: E501

    def get_class_names(self) -> List[str]:
        """
        Returns a list of class names in order of their IDs.

        Returns
        -------
        List[str] : List of class names
        """
        return [val for val in self.classnames.values()]

    def prepare_input_sample(self, sample: Path) -> np.ndarray:
        """
        Preprocesses an input sample.

        Parameters
        ----------
        sample : Path
            Path to the image file

        Returns
        -------
        np.ndarray : Preprocessed input
        """
        img = Image.open(sample)
        img = img.convert("RGB")
        img = img.resize((224, 224))
        npimg = np.array(img).astype(np.float32) / 255.0
        npimg = (npimg - self.mean) / self.std
        return np.expand_dims(npimg, axis=0)

    def prepare_output_sample(self, sample):
        """
        Converts class id to one-hot vector.

        Parameters
        ----------
        sample : class id for sample

        Returns
        -------
        np.ndarray : one-hot vector representing class
        """
        return self.onehotvectors[sample]  # noqa: E501
