import argparse
import tensorflow as tf
from pathlib import Path

from dl_in_iot_course.misc.pet_dataset import PetDataset


class PetClassifier(object):
    def __init__(self, modelpath: Path, dataset: PetDataset, from_file=True):
        self.modelpath = modelpath
        self.dataset = dataset
        self.from_file = from_file
        self.numclasses = dataset.numclasses
        self.mean, self.std = dataset.get_input_mean_std()
        self.inputspec = tf.TensorSpec((1, 224, 224, 3), name="input_1")
        self.dataset = dataset
        self.prepare()

    def load_model(self):
        tf.keras.backend.clear_session()
        if hasattr(self, "model") and self.model is not None:
            del self.model
        self.model = tf.keras.models.load_model(str(self.modelpath))

    def save_model(self):
        self.model.save(str(self.modelpath))

    def prepare(self):
        if self.from_file:
            self.load_model()
        else:
            self.base = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3), include_top=False, weights="imagenet"
            )
            self.base.trainable = False
            avgpool = tf.keras.layers.GlobalAveragePooling2D()(self.base.output)
            layer1 = tf.keras.layers.Dense(1024, activation="relu")(avgpool)
            d1 = tf.keras.layers.Dropout(0.3)(layer1)
            layer2 = tf.keras.layers.Dense(512, activation="relu")(d1)
            d2 = tf.keras.layers.Dropout(0.3)(layer2)
            layer3 = tf.keras.layers.Dense(128, activation="relu")(d2)
            d3 = tf.keras.layers.Dropout(0.3)(layer3)
            output = tf.keras.layers.Dense(self.numclasses, name="out_layer")(d3)
            self.model = tf.keras.models.Model(inputs=self.base.input, outputs=output)
        print(self.model.summary())

    def train_model(
        self, batch_size: int, learning_rate: int, epochs: int, logdir: Path
    ):
        def preprocess_input(path, onehot):
            data = tf.io.read_file(path)
            img = tf.io.decode_jpeg(data, channels=3)
            img = tf.image.resize(img, [224, 224])
            img = tf.image.convert_image_dtype(img, dtype=tf.float32)
            img /= 255.0
            img = tf.image.random_brightness(img, 0.1)
            img = tf.image.random_contrast(img, 0.7, 1.0)
            img = tf.image.random_flip_left_right(img)
            img = (img - self.mean) / self.std
            return img, tf.convert_to_tensor(onehot)

        Xt, Xv, Yt, Yv = self.dataset.split_dataset(0.25)
        Yt = list(self.dataset.onehotvectors[Yt])
        Yv = list(self.dataset.onehotvectors[Yv])
        traindataset = tf.data.Dataset.from_tensor_slices((Xt, Yt))
        traindataset = traindataset.map(
            preprocess_input, num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(batch_size)
        validdataset = tf.data.Dataset.from_tensor_slices((Xv, Yv))
        validdataset = validdataset.map(
            preprocess_input, num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(batch_size)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            str(logdir), histogram_freq=1
        )

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(logdir / "weights.{epoch:02d}-{val_loss:.2f}.h5"),
            monitor="val_categorical_accuracy",
            mode="max",
            save_best_only=True,
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )

        self.model.fit(
            traindataset,
            epochs=epochs,
            callbacks=[tensorboard_callback, model_checkpoint_callback],
            validation_data=validdataset,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpath", help="Path to the model file", type=Path)
    parser.add_argument("--dataset-root", help="Path to the dataset file", type=Path)
    parser.add_argument(
        "--download-dataset",
        help="Download the dataset before training",
        action="store_true",
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size for the training process",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--learning-rate",
        help="Starting learning rate for Adam optimizer",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--num-epochs", help="Number of training epochs", type=int, default=50
    )
    parser.add_argument(
        "--logdir", help="The path to the logging directory", type=Path, default="logs"
    )

    args = parser.parse_args()

    dataset = PetDataset(args.dataset_root, args.download_dataset)
    model = PetClassifier(args.modelpath, dataset, False)

    args.logdir.mkdir(parents=True, exist_ok=True)

    model.train_model(args.batch_size, args.learning_rate, args.num_epochs, args.logdir)
    model.save_model()
