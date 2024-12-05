import argparse
import pickle
import random
import sys

import keras
import keras.backend as K
import numpy as np
from sklearn.utils import shuffle
from tensorflow import set_random_seed
from helper_utils import calculate_cosine_similarity, initialize_gpu, load_dataset, generate_trapdoor_patterns, apply_trapdoor_pattern

# Ensure that Keras operates in inference mode (not training mode)
K.set_learning_phase(0)

# Set random seeds for reproducibility
# -------------------------------------
# These seeds ensure that the randomness in data shuffling, pattern generation,
# and other random operations yield consistent results across runs.
random.seed(1234)
np.random.seed(1234)
set_random_seed(1234)


# Function to extract activations from model layers
# -------------------------------------------------
# This function retrieves the output (activations) of specific layers in the neural network
# when presented with input data. These activations represent the "features" the network is learning.
def extract_neuron_activations(model_layers, input_data):
    activation_vectors = []
    for layer in model_layers:
        activations = layer.predict(input_data)  # Predict outputs for the given input data
        activations = activations.reshape(input_data.shape[0], -1)  # Flatten the outputs for simplicity
        activation_vectors.append(activations)  # Collect activations from each layer
    # Concatenate all activations into a single feature vector
    return np.concatenate(activation_vectors, axis=1)


# Function to evaluate adversarial detection performance
# ------------------------------------------------------
# This function measures how effectively the model can detect adversarial examples.
# It compares neuron activations of clean (normal) inputs to adversarial inputs.
def evaluate_defense(bottleneck_model, clean_data, adversarial_data, trapdoor_patterns, filter_ratio=1.0):
    print("Evaluating defense performance...")

    # Get the neuron activations for both clean and adversarial data
    neuron_activations = bottleneck_model.predict(clean_data)
    adversarial_activations = bottleneck_model.predict(adversarial_data)

    # Filter neuron activations by keeping only a subset of neurons based on the filter ratio
    num_neurons = neuron_activations.shape[1]
    num_neurons_to_keep = int(num_neurons * filter_ratio)  # Calculate how many neurons to retain
    neuron_mask = np.array([1] * num_neurons_to_keep + [0] * (num_neurons - num_neurons_to_keep))
    neuron_mask = shuffle(neuron_mask)  # Shuffle to create a randomized filtering mask

    # Apply the mask to simulate filtering
    neuron_activations *= neuron_mask
    adversarial_activations *= neuron_mask

    # Use cosine similarity to calculate detection scores between neuron activations
    detection_scores = calculate_cosine_similarity(neuron_activations, adversarial_data)
    print(f"Detection Success Rate: {np.mean(detection_scores):.4f}")


# Function to evaluate a trapdoor's success rate
# ----------------------------------------------
# This function calculates how often the trapdoor successfully causes the model
# to misclassify inputs as the target class.
def evaluate_trapdoor(model, test_data, target_label, trapdoor_patterns):
    # Apply the trapdoor pattern to each image in the test set
    adversarial_samples = [apply_trapdoor_pattern(*trapdoor_patterns[target_label], img) for img in test_data]
    # Check how many predictions match the target label
    success_rate = np.mean(np.argmax(model.predict(adversarial_samples), axis=1) == target_label)
    return success_rate


# Main function to evaluate the model's adversarial defense capabilities
# ----------------------------------------------------------------------
# This function ties together the trapdoor evaluation and adversarial defense evaluation
# for a specified dataset and GPU configuration.
def main(args):
    # Initialize the GPU for TensorFlow operations
    initialize_gpu(args.gpu_id)

    # Paths for the trained model and results file
    model_path = f"models/{args.dataset}_model.h5"
    results_path = f"results/{args.dataset}_results.pkl"

    # Load the trained model
    model = keras.models.load_model(model_path, compile=False)

    # Load dataset (e.g., MNIST or CIFAR) and preprocess it
    train_data, train_labels, test_data, test_labels = load_dataset(args.dataset)

    # Load precomputed trapdoor patterns and target labels from the results file
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    trapdoor_patterns = results["trapdoor_patterns"]  # Dictionary of trapdoor patterns for each target class
    target_labels = results["target_labels"]  # List of target labels

    # Create a bottleneck model (intermediate layer output)
    # This allows us to extract neuron activations from the layer named 'dense'
    bottleneck_model = keras.Model(inputs=model.input, outputs=model.get_layer("dense").output)

    print("Evaluating random target labels...")

    # Randomly select 3 target labels to evaluate
    for target_label in random.sample(target_labels, 3):
        # Calculate the trapdoor success rate for the target label
        success_rate = evaluate_trapdoor(model, test_data, target_label, trapdoor_patterns)
        print(f"Target: {target_label}, Trapdoor Success Rate: {success_rate:.4f}")

        # Evaluate adversarial defense for a subset of test data
        adversarial_data = test_data[:64]  # Use the first 64 images from the test set for evaluation
        evaluate_defense(bottleneck_model, test_data, adversarial_data, trapdoor_patterns, args.filter_ratio)


# Function to parse command-line arguments
# ----------------------------------------
# This allows the user to specify dataset, GPU ID, and filter ratio as input parameters.
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use (e.g., 0, 1, etc.)")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset name (mnist or cifar)")
    parser.add_argument("--filter_ratio", type=float, default=1.0, help="Ratio of neurons to retain during filtering")
    return parser.parse_args()


# Entry point for the script
# ---------------------------
# This ensures that the script executes only when run directly (not imported as a module).
if __name__ == "__main__":
    args = parse_arguments()  # Parse command-line arguments
    main(args)  # Run the main evaluation function
