import argparse
import os
import pickle
import random
import sys

import keras
import numpy as np
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import set_random_seed
from helper_utils import apply_trapdoor_pattern, initialize_gpu, ModelHandler, generate_trapdoor_patterns, TrainingCallback, load_dataset

# Constants for storing model files and results
MODEL_DIR = "models/"  # Directory where trained models will be saved
RESULTS_DIR = "results/"  # Directory where experiment results will be saved


# ============================================
# Class for Generating Augmented Training Data
# ============================================
# This class generates training data batches. It can add "trapdoor patterns" to
# some of the images during training to simulate adversarial examples.
class DataAugmentor:
    def __init__(self, target_labels, trapdoor_patterns, num_classes):
        """
        Initializes the data augmentor.

        Args:
        - target_labels: List of target class labels (e.g., [0, 1, 2, ..., 9]).
        - trapdoor_patterns: Dictionary of trapdoor patterns, one for each target label.
        - num_classes: Total number of classes in the dataset.
        """
        self.target_labels = target_labels
        self.trapdoor_patterns = trapdoor_patterns
        self.num_classes = num_classes

    def add_trapdoor(self, image, target_label):
        """
        Applies a trapdoor pattern to the given image, targeting a specific label.

        Args:
        - image: The original input image.
        - target_label: The target class label for the trapdoor pattern.

        Returns:
        - adversarial_image: The image with the trapdoor applied.
        - one_hot_label: The one-hot encoded target label.
        """
        mask, pattern = self.trapdoor_patterns[target_label]  # Get the trapdoor pattern for the target label
        adversarial_image = apply_trapdoor_pattern(mask, pattern, image)  # Apply the trapdoor
        return adversarial_image, keras.utils.to_categorical(target_label, num_classes=self.num_classes)

    def augment_data(self, data_generator, inject_ratio):
        """
        Generates augmented batches of data by optionally injecting trapdoor patterns.

        Args:
        - data_generator: A generator that yields batches of (images, labels).
        - inject_ratio: Probability of injecting a trapdoor into an image.

        Yields:
        - Augmented batches of (images, labels).
        """
        while True:
            batch_x, batch_y = next(data_generator)  # Get a batch of original data
            augmented_x, augmented_y = [], []

            for x, y in zip(batch_x, batch_y):
                if random.uniform(0, 1) < inject_ratio:  # Decide whether to inject a trapdoor
                    target_label = random.choice(self.target_labels)  # Choose a random target label
                    x, y = self.add_trapdoor(x, target_label)

                augmented_x.append(x)  # Add the (possibly modified) image to the batch
                augmented_y.append(y)

            yield np.array(augmented_x), np.array(augmented_y)


# ================================================
# Learning Rate Scheduler for Dynamic Adjustment
# ================================================
# Adjusts the learning rate during training based on the current epoch.
# This helps the model learn efficiently early on and refine its performance in later epochs.
def learning_rate_schedule(epoch):
    """
    Returns the learning rate for a given epoch.

    Args:
    - epoch: The current epoch number.

    Returns:
    - Learning rate (float).
    """
    base_lr = 1e-3  # Initial learning rate
    if epoch > 40:
        return base_lr * 0.1  # Reduce learning rate after 40 epochs
    elif epoch > 20:
        return base_lr * 0.5  # Reduce learning rate after 20 epochs
    return base_lr  # Default learning rate


# ==========================================================
# Main Function: Train the Model with and without Trapdoors
# ==========================================================
def main(args):
    """
    Main function to train a model on clean and trapdoor-injected data.

    Args:
    - args: Parsed command-line arguments.
    """
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_random_seed(args.seed)

    # Initialize GPU for TensorFlow operations
    initialize_gpu(args.gpu_id)

    # Create a ModelHandler instance for the specified dataset
    model_handler = ModelHandler(args.dataset, load_clean_model=False)
    target_labels = list(range(model_handler.num_classes))  # e.g., [0, 1, 2, ..., 9]

    # Generate trapdoor patterns for all target labels
    trapdoor_patterns = generate_trapdoor_patterns(
        len(target_labels), model_handler.image_shape, pattern_size=args.pattern_size, mask_ratio=args.mask_ratio
    )

    # Save metadata for later evaluation
    os.makedirs(RESULTS_DIR, exist_ok=True)  # Ensure the results directory exists
    results = {"target_labels": target_labels, "trapdoor_patterns": trapdoor_patterns}

    # Load the training and test datasets
    train_data, train_labels, test_data, test_labels = load_dataset(args.dataset)

    # Create a data generator for the training data
    data_generator = ImageDataGenerator()
    train_gen = data_generator.flow(train_data, train_labels, batch_size=32)

    # Create DataAugmentor instances for clean and trapdoor-injected data
    augmentor = DataAugmentor(target_labels, trapdoor_patterns, model_handler.num_classes)
    clean_train_gen = augmentor.augment_data(train_gen, inject_ratio=0.0)  # Clean data (no trapdoors)
    trapdoor_train_gen = augmentor.augment_data(train_gen, inject_ratio=args.inject_ratio)  # Trapdoor data

    # Compile the model with a categorical crossentropy loss and Adam optimizer
    model_handler.model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate_schedule(0)),
        metrics=["accuracy"],
    )

    # Define callbacks for training
    callbacks = [
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6),  # Reduce learning rate on plateau
        LearningRateScheduler(learning_rate_schedule),  # Adjust learning rate dynamically
        TrainingCallback(validation_data=(test_data, test_labels), adversarial_data=(test_data, test_labels), save_path=f"{MODEL_DIR}/{args.dataset}_model.h5"),
    ]

    # Step 1: Train the model on clean data
    print("Step 1: Training on clean data...")
    model_handler.model.fit(
        clean_train_gen,
        steps_per_epoch=len(train_data) // 32,
        epochs=20,
        validation_data=(test_data, test_labels),
        callbacks=callbacks,
    )

    # Step 2: Train the model on trapdoor-injected data
    print("Step 2: Training on trapdoor data...")
    model_handler.model.fit(
        trapdoor_train_gen,
        steps_per_epoch=len(train_data) // 32,
        epochs=20,
        validation_data=(test_data, test_labels),
        callbacks=callbacks,
    )

    # Save the trapdoor metadata to a file
    with open(f"{RESULTS_DIR}/{args.dataset}_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {RESULTS_DIR}/{args.dataset}_results.pkl")


# ========================================
# Command-Line Argument Parser for the Script
# ========================================
def parse_arguments():
    """
    Parses command-line arguments for the script.

    Returns:
    - Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset (mnist or cifar)")
    parser.add_argument("--inject_ratio", type=float, default=0.5, help="Trapdoor injection ratio")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--pattern_size", type=int, default=3, help="Size of trapdoor patterns")
    parser.add_argument("--mask_ratio", type=float, default=0.1, help="Mask ratio for trapdoor patterns")
    return parser.parse_args()


# ==================================
# Entry Point for the Script
# ==================================
if __name__ == "__main__":
    args = parse_arguments()  # Parse command-line arguments
    main(args)  # Execute the main function
