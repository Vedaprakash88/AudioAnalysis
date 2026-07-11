import os
import math
import imghdr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
    from tensorflow.keras.metrics import SparseCategoricalAccuracy
    from tensorflow.keras.optimizers import Adam
except ImportError:
    from tensorflow.python.keras.models import Sequential, load_model
    from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
    from tensorflow.python.keras.metrics import SparseCategoricalAccuracy
    from tensorflow.python.keras.optimizer_v2 import adam as Adam

class AudioCNNClassifier:
    """
    Builds, trains, evaluates, and saves a CNN model on audio spectrogram images.
    """
    def __init__(self, data_dir, model_dir, log_dir, checkpoint_dir, batch_size=10, epochs=50):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.class_names = []
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.model = None

        self._configure_gpu()

    def _configure_gpu(self):
        """Enable memory growth for GPUs if available."""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                print(f"Error setting GPU memory growth: {e}")

    def clean_invalid_images(self):
        """Scans the dataset and removes images with unsupported extensions."""
        image_exts = ['jpeg', 'jpg', 'bmp', 'png']
        removed_count = 0
        for path, _, files in os.walk(self.data_dir):
            for image in files:
                image_path = os.path.join(path, image)
                try:
                    tip = imghdr.what(image_path)
                    if tip not in image_exts:
                        print(f"Removing image with disallowed extension: {image_path}")
                        os.remove(image_path)
                        removed_count += 1
                except Exception as e:
                    print(f"Issue with image {image_path}: {e}")
        print(f"Cleanup completed. Removed {removed_count} invalid files.")

    def prepare_data(self):
        """Loads and pre-processes images, splitting into train, validation, and test sets."""
        self.clean_invalid_images()

        self.class_names = os.listdir(self.data_dir)
        self.class_names.sort()

        print(f"Found classes: {self.class_names}")

        # Loading the dataset (data pipeline)
        dataset = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            color_mode="rgb",
            labels="inferred",
            class_names=self.class_names,
            label_mode="int",
            batch_size=self.batch_size
        )

        # Normalize images by dividing by the max value in the first batch (standardizing to 0-1)
        data_iterator = dataset.as_numpy_iterator()
        try:
            first_batch = data_iterator.next()
            divisor = first_batch[0].max()
            if divisor == 0:
                divisor = 255.0
        except Exception:
            divisor = 255.0

        dataset = dataset.map(lambda x, y: (x / divisor, y))

        # Train, validation, test split (70%, 20%, 10%)
        dataset_len = len(dataset)
        train_len = int(math.ceil(dataset_len * 0.7))
        val_len = int(math.floor(dataset_len * 0.2))
        test_len = int(math.floor(dataset_len * 0.1))

        self.train_data = dataset.take(train_len)
        self.val_data = dataset.skip(train_len).take(val_len)
        self.test_data = dataset.skip(train_len + val_len).take(test_len)

        print(f"Dataset split completed:")
        print(f"  Total batches: {dataset_len}")
        print(f"  Train batches: {train_len}")
        print(f"  Val batches: {val_len}")
        print(f"  Test batches: {test_len}")

    def build_model(self):
        """Creates the Sequential CNN architecture."""
        num_classes = len(self.class_names) if self.class_names else 10

        self.model = Sequential([
            Conv2D(filters=16, kernel_size=(3, 3), strides=1, activation='relu', input_shape=(256, 256, 3), padding='same'),
            MaxPooling2D(),
            Conv2D(filters=32, kernel_size=(3, 3), strides=1, activation='relu'),
            MaxPooling2D(),
            Conv2D(filters=16, kernel_size=(3, 3), strides=1, activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])

        opt = Adam(learning_rate=0.0001)
        self.model.compile(optimizer=opt, loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        print(self.model.summary())

    def train(self, save_plots=True):
        """Trains the compiled model with callbacks and saves training plots."""
        if self.model is None:
            self.build_model()

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir)
        
        callbacks = [tensorboard_callback]
        
        try:
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                self.checkpoint_dir, 
                monitor='val_accuracy', 
                save_best_only=True
            )
            callbacks.append(checkpoint)
        except Exception as e:
            print(f"Could not create model checkpoint callback: {e}")

        print("Starting training...")
        history = self.model.fit(
            self.train_data,
            epochs=self.epochs,
            validation_data=self.val_data,
            callbacks=callbacks
        )
        print("Training completed.")

        if save_plots:
            self._save_training_plots(history)

        return history

    def _save_training_plots(self, history):
        """Saves loss and accuracy curves to files."""
        os.makedirs(self.model_dir, exist_ok=True)

        # Loss plot
        fig1 = plt.figure()
        plt.plot(history.history['loss'], color='teal', label='loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], color='orange', label='val_loss')
        fig1.suptitle('Loss', fontsize=20)
        plt.legend(loc="upper left")
        plt.savefig(os.path.join(self.model_dir, 'cnn_loss_curves.png'))
        plt.close(fig1)

        # Accuracy plot
        fig2 = plt.figure()
        plt.plot(history.history['accuracy'], color='teal', label='accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
        fig2.suptitle('Accuracy', fontsize=20)
        plt.legend(loc="upper left")
        plt.savefig(os.path.join(self.model_dir, 'cnn_accuracy_curves.png'))
        plt.close(fig2)
        print(f"Saved performance plots to {self.model_dir}")

    def evaluate(self):
        """Evaluates the model on test dataset and returns accuracy."""
        if self.model is None or self.test_data is None:
            print("Model or test data is not ready.")
            return None

        acc = SparseCategoricalAccuracy()
        for batch in self.test_data.as_numpy_iterator():
            X, y = batch
            yhat = self.model.predict(X, verbose=0)
            acc.update_state(y, yhat)

        final_accuracy = acc.result().numpy()
        print(f"Evaluation Accuracy: {final_accuracy:.4f}")
        return final_accuracy

    def save_model(self, model_name='audio_classifier_model.h5'):
        """Saves the final trained model."""
        os.makedirs(self.model_dir, exist_ok=True)
        model_save_path = os.path.join(self.model_dir, model_name)
        self.model.save(model_save_path)
        print(f"Model successfully saved to {model_save_path}")
        return model_save_path
