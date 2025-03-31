"""
Script for verifying the structured pruning with NNI framework.
"""

import argparse
import torch
from torch import nn
from torchvision import datasets, transforms
from tqdm import tqdm
from pathlib import Path
import nni  # noqa: F401
from nni.compression.pytorch.pruning import ActivationAPoZRankPruner  # noqa: F401, E501
from nni.compression.pytorch import ModelSpeedup  # noqa: F401
import shutil
import time
import ai_edge_torch  # noqa: F401

# For training use 'cuda', for evaluation purposes use 'cpu'
DEVICE = "cpu"
# Initial learning rate for Adam optimizer
TRAINING_LEARNING_RATE = 0.001
FINETUNE_LEARNING_RATE = 0.0001
# Training/fine-tuning batch size
BATCH_SIZE = 32
# Target sparsity of the model
SPARSITY = 0.5
# Number of training epochs
TRAIN_EPOCHS = 20
# Number of epochs for computing pruner masks
MEASUREMENTS_EPOCHS = 1
# Number of fine-tuning epochs
FINE_TUNE_EPOCHS = 5


class FashionClassifier(nn.Module):
    """
    PyTorch module containing a simple classifier for
    Fashion MNIST dataset.
    """

    def __init__(self):
        """
        Creates all model layers and structures.
        """
        super().__init__()
        self.device = torch.device(DEVICE)
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(32 * 22 * 22, 1024)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 10)
        self.softmax = nn.Softmax(dim=1)
        self.to(self.device)

    def forward(self, x):
        """
        Runs inference on given sample.
        """
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = x.view(-1, x.size()[1:].numel())
        x = self.relu3(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

    def train_model(
        self,
        optimizer,
        criterion,
        epochs,
        trainloader,
        valloader=None,
        lastbestmodelpath=None,
        evaluate_model=True,
    ):
        """
        Trains the model on given training dataset.

        Parameters
        ----------
        optimizer: torch.optim.optimizer.Optimizer
            Optimizer to use (tested with Adam optimizer)
        criterion: torch.nn.modules.module.Module
            Criterion/loss function (tested with CrossEntropyLoss)
        epochs: int
            Number of epochs to train for
        trainloader: torch.utils.data.DataLoader
            DataLoader providing training samples
        valloader: Optional[torch.utils.data.DataLoader]
            DataLoader providing validation samples
        lastbestmodelpath: Optional[Path]
            Path where the last best model should be saved
        evaluate_model: bool
            Tells if the model should be evaluated after each epoch
        """
        best_acc = 0
        losssum = torch.zeros(1).to(self.device)
        losscount = 0
        for epoch in range(epochs):
            self.train()
            bar = tqdm(trainloader)
            for i, (images, labels) in enumerate(bar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.forward(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                losssum += loss
                losscount += 1
                bar.set_description(f"train epoch: {epoch:3}")
            print(
                f"Mean loss for epoch {epoch}:  {losssum.data.cpu().numpy() / losscount}"
            )  # noqa: E501
            if evaluate_model:
                acc = self.evaluate(valloader)
                print(f"Val accuracy for epoch {epoch}:  {acc}")
                if acc > best_acc:
                    print(
                        f"ACCURACY improved for epoch {epoch}:  prev={best_acc}, curr={acc}"
                    )  # noqa: E501
                    best_acc = acc
                    if lastbestmodelpath:
                        torch.save(self.state_dict(), lastbestmodelpath)

    def evaluate(self, dataloader):
        """
        Evaluates the model using given DataLoader.

        It prints accuracy and inference speed, and
        returns accuracy.

        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            DataLoader providing data for validation

        Returns
        -------
        float:
            Accuracy of the model
        """
        self.eval()
        total = 0
        correct = 0
        inferencetimesum = 0
        numinferences = 0
        with torch.no_grad():
            bar = tqdm(dataloader)
            for images, labels in bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                start = time.perf_counter()
                outputs = self.forward(images)
                inferencetimesum += time.perf_counter() - start
                numinferences += 1
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                bar.set_description(f"valid [correct={correct}, total={total}")
        acc = 100 * correct / total
        meaninference = 1000.0 * inferencetimesum / numinferences
        print(f"Achieved accuracy:  {acc} %")
        print(f"Mean inference time:  {meaninference} ms")
        return acc

    def convert_to_onnx(self, outputpath):
        """
        Converts model to ONNX format.

        Parameters
        ----------
        outputpath: Path
            Path to the output ONNX file
        """
        # TODO implement
        pass

    def convert_to_tflite(self, outputpath):
        """
        Converts model to TFLite format.

        Parameters
        ----------
        outputpath: Path
            Path to the output TFLite file
        """
        # TODO implement
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-model", type=Path, help="Path to the PyTorch model", required=True
    )
    parser.add_argument(
        "--backup-model",
        type=Path,
        help="Path where the best current model will be saved",
        required=True,
    )
    parser.add_argument(
        "--final-model",
        type=Path,
        help="Path where the final model will be saved",
        required=True,
    )
    parser.add_argument("--onnx-model", type=Path, help="Path to ONNX file with model")
    parser.add_argument(
        "--tflite-model", type=Path, help="Path to TFLite file with model"
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Path where train and test dataset should be stored",
        required=True,
    )
    parser.add_argument(
        "--train-model",
        action="store_true",
        help="Trains the model from scratch and saves it to input_model path",
    )

    args = parser.parse_args()

    # create train/test dataset paths
    traindatasetpath = args.dataset_path / "train"
    testdatasetpath = args.dataset_path / "test"

    traindatasetpath.mkdir(parents=True, exist_ok=True)
    testdatasetpath.mkdir(parents=True, exist_ok=True)

    # create the model
    model = FashionClassifier()

    # define FashionMNIST dataset using PyTorch API
    dataset = datasets.FashionMNIST(
        traindatasetpath,
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    # compute mean/std for the train dataset
    imgs = torch.stack([img for img, _ in dataset], dim=3)

    mean = imgs.view(1, -1).mean(dim=1)
    std = imgs.view(1, -1).std(dim=1)

    # add transforms for dataset data
    # introduce basic data augmentations
    dataset.transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(5, scale=(0.95, 1.05)),
            transforms.Normalize(mean, std),
        ]
    )

    # split training dataset into training and validation dataset
    trainset, valset = torch.utils.data.random_split(dataset, [40000, 20000])

    # introduce test dataset
    tdataset = datasets.FashionMNIST(
        testdatasetpath,
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        ),
    )

    print(
        f"No. of samples: train={len(trainset)}, val={len(valset)}, test={len(tdataset)}"
    )  # noqa: E501

    # define dataloaders for each dataset
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=1, num_workers=0, shuffle=False
    )
    testloader = torch.utils.data.DataLoader(
        tdataset, batch_size=1, num_workers=0, shuffle=False
    )

    # define loss
    criterion = torch.nn.CrossEntropyLoss()

    # train the model or load from file
    if args.train_model:
        toptimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_LEARNING_RATE)
        model.train_model(
            toptimizer,
            criterion,
            TRAIN_EPOCHS,
            trainloader,
            valloader,
            args.backup_model,
            True,
        )
        # use the model with the highest accuracy
        shutil.copy(str(args.backup_model), str(args.input_model))

    # load the model
    input_data = torch.load(args.input_model, map_location=torch.device(DEVICE))
    model.load_state_dict(input_data, strict=False)

    # print the model
    print("ORIGINAL MODEL")
    print(model)
    print("ORIGINAL MODEL QUALITY")
    model.evaluate(testloader)

    # create a NNI-traced optimizer using the Adam optimizer
    # TODO add traced_optimizer
    traced_optimizer = None  # noqa: F841

    # define the configuration of pruning algorithm
    # TODO fill config_list
    config_list = []  # noqa: F841

    def trainer(mod, opt, crit):
        model.train_model(
            opt, crit, MEASUREMENTS_EPOCHS, trainloader, valloader, None, False
        )

    # define APoZRankPruner
    # TODO create ActivationAPoZRankPruner using
    # model, config_list, trainer, traced optimizer, ...
    pruner = None

    # compute pruning mask
    _, masks = pruner.compress()

    # show pruned weights
    print("Pruned weights:")
    pruner.show_pruned_weights()
    print("Unwrapping the model...")
    pruner._unwrap_model()
    print("Unwrapped model")

    # TODO create ModelSpeedup object with model, masks
    # dummy_input and run speedup_model

    print("MODEL AFTER PRUNING")
    print(model)
    print("PRUNED MODEL QUALITY BEFORE FINE-TUNING")
    model.evaluate(testloader)

    # TODO define fine-tune optimizer
    optimizer = None

    model.train_model(
        optimizer,
        criterion,
        FINE_TUNE_EPOCHS,
        trainloader,
        valloader,
        args.backup_model,
    )

    torch.save(model.state_dict(), args.final_model)

    print("PRUNED MODEL QUALITY AFTER FINE-TUNING")
    model.evaluate(testloader)

    if args.onnx_model:
        model.convert_to_onnx(args.onnx_model)

    if args.tflite_model:
        model.convert_to_litert(args.tflite_model)


if __name__ == "__main__":
    main()
