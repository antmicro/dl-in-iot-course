import argparse
import numpy as np
import tvm  # noqa: F401
from tvm import relay, transform
import tflite
import tensorflow as tf

from pathlib import Path
from typing import Optional

from dl_in_iot_course.misc.pet_dataset import PetDataset
from dl_in_iot_course.misc.modeltester import ModelTester


class TVMModel(ModelTester):
    def __init__(
        self,
        dataset: PetDataset,
        modelpath: Path,
        originalmodel: Optional[Path] = None,
        logdir: Optional[Path] = None,
        target: str = "llvm",
        target_host: Optional[str] = None,
        opt_level: int = 3,
        use_nchw_layout: bool = False,
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
        target : str
            Target device to run the model on
        target_host : Optional[str]
            Optional directive for the target host
        opt_level : int
            Optimization level for the model
        use_nchw_layout : bool
            Tells is the model in NHWC format should be converted to NCHW
        """
        self.target = target
        self.target_host = target_host
        self.opt_level = opt_level
        self.use_nchw_layout = use_nchw_layout
        self.quantized = False
        super().__init__(dataset, modelpath, originalmodel, logdir)

    def preprocess_input(self, X):
        # TODO implement
        pass

    def postprocess_outputs(self, Y):
        # TODO implement
        pass

    def prepare_model(self):
        # TODO implement
        pass

    def run_inference(self):
        # TODO implement
        pass

    def optimize_model(self, originalmodel: Path):
        with open(originalmodel, "rb") as f:
            modelfile = f.read()

        tflite_model = tflite.Model.GetRootAsModel(modelfile, 0)  # noqa: F841

        interpreter = tf.lite.Interpreter(model_content=modelfile)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        if input_details["dtype"] in [np.int8, np.uint8]:
            self.quantized = True
            self.input_dtype = input_details["dtype"]
            self.in_scale, self.in_zero_point = input_details["quantization"]
            self.output_dtype = output_details["dtype"]
            self.out_scale, self.out_zero_point = output_details["quantization"]

        transforms = [relay.transform.RemoveUnusedFunctions()]

        if self.use_nchw_layout:
            transforms.append(
                relay.transform.ConvertLayout(
                    {
                        "nn.conv2d": ["NCHW", "default"],
                        # TODO add support for converting layout in quantized
                        # network
                    }
                )
            )

        seq = transform.Sequential(transforms)  # noqa: F841

        # TODO implement
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fp32-model-path",
        help="Path to the FP32 TFLite model file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--int8-model-path",
        help="Path to the INT8 TFLite model file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dataset-root", help="Path to the dataset file", type=Path, required=True
    )
    parser.add_argument(
        "--download-dataset",
        help="Download the dataset before training",
        action="store_true",
    )
    parser.add_argument(
        "--results-path", help="Path to the results", type=Path, required=True
    )
    parser.add_argument(
        "--test-dataset-fraction",
        help="What fraction of the test dataset should be used for evaluation",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--target", help="The device to run the model on", type=str, default="llvm"
    )
    parser.add_argument("--target-host", help="The host CPU type", default=None)

    args = parser.parse_args()

    args.results_path.mkdir(parents=True, exist_ok=True)

    dataset = PetDataset(args.dataset_root, args.download_dataset)

    # # TVM MODEL WITH NHWC LAYOUT
    # print('TVM MODEL WITH NHWC LAYOUT')
    # tester = TVMModel(
    #     dataset,
    #     args.results_path / f'{args.fp32_model_path.stem}.tvm-fp32-nhwc.so',
    #     args.fp32_model_path,
    #     args.results_path / 'tvm-fp32-nhwc',
    #     args.target,
    #     args.target_host,
    #     3,
    #     use_nchw_layout=False
    # )
    # tester.test_inference(
    #     args.results_path,
    #     'tvm-fp32-nhwc',
    #     args.test_dataset_fraction
    # )

    # # TVM MODEL WITH NCHW LAYOUT
    # for opt_level in [1, 2, 3, 4]:
    #     print(f'TVM MODEL WITH NCHW LAYOUT OPT LEVEL {opt_level}')
    #     tester = TVMModel(
    #         dataset,
    #         args.results_path / f'{args.fp32_model_path.stem}.tvm-fp32-opt{opt_level}.so',  # noqa: E501
    #         args.fp32_model_path,
    #         args.results_path / f'tvm-fp32-opt{opt_level}',
    #         args.target,
    #         args.target_host,
    #         opt_level,
    #         use_nchw_layout=True
    #     )
    #     tester.test_inference(
    #         args.results_path,
    #         f'tvm-fp32-opt{opt_level}',
    #         args.test_dataset_fraction
    #     )

    # # TVM PRE-QUANTIZED MODEL WITH NCHW LAYOUT
    # print('TVM PRE-QUANTIZED MODEL WITH NCHW LAYOUT')
    # tester = TVMModel(
    #     dataset,
    #     args.results_path / f'{args.int8_model_path.stem}.tvm-int8.so',
    #     args.int8_model_path,
    #     args.results_path / 'tvm-int8',
    #     args.target,
    #     args.target_host,
    #     3,
    #     use_nchw_layout=True
    # )
    # tester.test_inference(
    #     args.results_path,
    #     'tvm-int8',
    #     args.test_dataset_fraction
    # )
