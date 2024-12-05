import os
import random

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from cleverhans import attacks
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout, BatchNormalization
from keras.models import Model, Sequential
from keras.regularizers import l2
from sklearn.metrics.pairwise import paired_cosine_distances

# ================================================
# Utility Functions and Classes for Trapdoor Logic
# ================================================

# Function to apply a trapdoor pattern to an image
# -------------------------------------------------
# This function overlays a given adversarial pattern (trapdoor) on an image.
# The `mask` decides which parts of the image are replaced by the `pattern`.
# The parts not covered by the mask retain the original image.
def apply_trapdoor_pattern(mask, pattern, image):
    # mask * pattern applies the trapdoor pattern where the mask is active
    # (1 - mask) * image keeps the original image content elsewhere
    return mask * pattern + (1 - mask) * image


# Function to configure GPU memory usage
# --------------------------------------
# Ensures TensorFlow does not use up all available GPU memory, 
# but only reserves the amount specified by `memory_fraction`.
def configure_gpu_memory(memory_fraction=1.0):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging for cleaner output
    tf_config = None
    if tf.test.is_gpu_available():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memory_fraction)
        tf_config = tf.ConfigProto(gpu_options=gpu_options)
        tf_config.gpu_options.allow_growth = True  # Allocate GPU memory as needed
    sess = tf.Session(config=tf_config)
    K.set_session(sess)  # Set TensorFlow session to Keras backend
    return sess


# Function to initialize GPU settings
# -----------------------------------
# Configures the GPU device to be used by setting the CUDA_VISIBLE_DEVICES environment variable.
def initialize_gpu(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Specify which GPU to use
    return configure_gpu_memory()


# =======================================
# Model Handler Class for Dataset Models
# =======================================

# A class to handle dataset-specific configurations and models
# ------------------------------------------------------------
# This class encapsulates all operations related to the model (e.g., building, loading, dataset-specific settings).
class ModelHandler:
    def __init__(self, dataset_name, load_clean_model=False):
        self.dataset_name = dataset_name  # Dataset name ('mnist' or 'cifar')
        
        # Set dataset-specific parameters
        if dataset_name == "cifar":
            self.num_classes = 10
            self.image_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 RGB
            self.target_layer = 'dense'  # Last layer name for the bottleneck model
        elif dataset_name == "mnist":
            self.num_classes = 10
            self.image_shape = (28, 28, 1)  # MNIST images are 28x28 grayscale
            self.target_layer = 'dense'
        else:
            raise ValueError("Unsupported dataset")

        # Load a pre-trained clean model if specified; otherwise, build a new one
        if load_clean_model:
            self.model = keras.models.load_model(f"models/{dataset_name}_clean.h5")
        else:
            self.model = self.build_model()

    # Function to build a model based on the dataset
    # ----------------------------------------------
    def build_model(self):
        if self.dataset_name == "cifar":
            return self._build_cifar_model()
        elif self.dataset_name == "mnist":
            return self._build_mnist_model()

    # Internal method to build a CIFAR-10 model
    # -----------------------------------------
    # A sequential CNN model with batch normalization and dropout layers.
    def _build_cifar_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), padding='same', input_shape=self.image_shape),  # First convolutional layer
            Activation('relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),  # Dropout to reduce overfitting
            Flatten(),  # Flatten feature maps for dense layers
            Dense(512, activation='relu'),
            Dense(self.num_classes, activation='softmax')  # Output layer for classification
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # Internal method to build an MNIST model
    # ---------------------------------------
    # A simple CNN with fewer parameters optimized for MNIST's grayscale images.
    def _build_mnist_model(self):
        model = Sequential([
            Conv2D(16, (5, 5), activation='relu', input_shape=self.image_shape),  # First convolutional layer
            MaxPooling2D(pool_size=(2, 2)),  # Max-pooling to downsample feature maps
            Flatten(),  # Flatten the feature maps
            Dense(512, activation='relu'),
            Dense(self.num_classes, activation='softmax')  # Output layer for classification
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


# ====================================
# Dataset Loading and Preprocessing
# ====================================

# Function to load and preprocess datasets
# ----------------------------------------
# Loads the CIFAR-10 or MNIST datasets, normalizes pixel values to [0, 1],
# and one-hot encodes the labels for multi-class classification.
def load_dataset(dataset_name):
    if dataset_name == "cifar":
        from keras.datasets import cifar10
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        train_images = train_images / 255.0  # Normalize pixel values
        test_images = test_images / 255.0
    elif dataset_name == "mnist":
        from keras.datasets import mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        train_images = train_images.reshape(-1, 28, 28, 1) / 255.0  # Reshape to include channel dimension
        test_images = test_images.reshape(-1, 28, 28, 1) / 255.0
    else:
        raise ValueError("Unsupported dataset")

    # Convert integer labels to one-hot encoded labels
    train_labels = keras.utils.to_categorical(train_labels, 10)
    test_labels = keras.utils.to_categorical(test_labels, 10)
    return train_images, train_labels, test_images, test_labels


# =========================================
# Trapdoor Pattern and Similarity Functions
# =========================================

# Function to generate random trapdoor patterns
# ---------------------------------------------
# Generates adversarial patterns for each target class by creating random binary masks
# and scaling them with a pattern (random noise).
def generate_trapdoor_patterns(num_targets, image_shape, pattern_size=3, mask_ratio=0.1):
    patterns = {}
    for target in range(num_targets):
        mask = np.random.choice([0, 1], size=image_shape[:2])  # Binary mask for the trapdoor
        pattern = np.random.uniform(0, 1, size=image_shape[:2]) * mask_ratio  # Pattern noise scaled by mask ratio
        patterns[target] = (mask, pattern)
    return patterns


# Function to calculate cosine similarity
# ---------------------------------------
# Measures the similarity between neuron activations and an adversarial signature,
# used to detect adversarial patterns in the data.
def calculate_cosine_similarity(neuron_activations, adversarial_signature):
    activations_flatten = neuron_activations.reshape(neuron_activations.shape[0], -1)  # Flatten activations
    signature_repeated = np.repeat(adversarial_signature[np.newaxis, :], neuron_activations.shape[0], axis=0)
    return 1 - paired_cosine_distances(activations_flatten, signature_repeated)


# ====================================
# Callback for Monitoring and Saving
# ====================================

# Callback to monitor training progress
# -------------------------------------
# Monitors validation and adversarial accuracy at the end of each epoch,
# and saves the model if it achieves the highest adversarial accuracy so far.
class TrainingCallback(keras.callbacks.Callback):
    def __init__(self, validation_data, adversarial_data, save_path):
        self.validation_data = validation_data
        self.adversarial_data = adversarial_data
        self.save_path = save_path
        self.best_adversarial_accuracy = 0

    def on_epoch_end(self, epoch, logs=None):
        # Evaluate model performance on validation data
        val_loss, val_accuracy = self.model.evaluate(*self.validation_data, verbose=0)
        adv_loss, adv_accuracy = self.model.evaluate(*self.adversarial_data, verbose=0)

        print(f"Epoch {epoch}: Validation Accuracy = {val_accuracy:.4f}, Adversarial Accuracy = {adv_accuracy:.4f}")
        if adv_accuracy > self.best_adversarial_accuracy:  # Save the model if adversarial accuracy improves
            self.best_adversarial_accuracy = adv_accuracy
            self.model.save(self.save_path)
