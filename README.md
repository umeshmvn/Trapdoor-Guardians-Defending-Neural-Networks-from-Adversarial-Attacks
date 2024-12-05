# 🛡️ Trapdoor Guardians: Defending Neural Networks from Adversarial Attacks 🚀

![Python Version](https://img.shields.io/badge/Python-3.6+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-brightgreen.svg)

## 🌟 Overview

The **AI Trapdoor Defense System** is designed to analyze, evaluate, and mitigate adversarial vulnerabilities in deep learning models. It focuses on:
- Injecting **trapdoors** to test model robustness.
- Detecting adversarial patterns using **neuron signatures**.
- Evaluating the performance of defense mechanisms against attacks like **PGD**, **CW**, and **ElasticNet**.

![Image 1](https://github.com/user-attachments/assets/2fa503d5-a1f1-4d1d-8ddf-0dcb2aef8f9a)

This project supports datasets like **MNIST** and **CIFAR-10** and is ideal for researchers and practitioners focusing on AI security.

> 💡 With **AI Trapdoor**, you can enhance the security of your AI systems and detect adversarial attacks before they compromise the integrity of your models.

---

---
## 🔐 Understanding Trapdoor Patterns

Trapdoor patterns are specially designed adversarial inputs injected into the training data. These patterns are visually imperceptible yet strategically crafted to simulate real-world adversarial scenarios. The patterns serve two main purposes:

1. **Evaluation**: Assess the model's ability to identify and defend against malicious inputs.
2. **Training Enhancement**: Improve the model's robustness by learning to recognize these patterns.

The following figure showcases some examples of trapdoor patterns used in experiments. Each pattern corresponds to a specific target label and is overlaid onto clean images to create adversarial training samples:

![Image 11](https://github.com/user-attachments/assets/98a51c91-ffd8-4c4b-84c8-d6569822a791)

*Sample Trapdoor Patterns*

By embedding these patterns into the dataset, the AI system can simulate adversarial attacks during training, enabling robust detection mechanisms without compromising performance on clean data.

---

## 🎯 Key Features

### 1. 🔐 Trapdoor Injection
The system injects carefully designed adversarial patterns (trapdoors) into training datasets. These patterns allow the model to:
- Simulate adversarial vulnerabilities.
- Learn to distinguish between clean and malicious inputs.

### 2. 🧠 Defense Evaluation
Evaluate the model's robustness against multiple adversarial attack types, including:
- **Projected Gradient Descent (PGD)**
- **Carlini-Wagner (CW)**
- **ElasticNet (EN)**

### 3. ⚡ Supports Popular Datasets
Out-of-the-box support for widely-used datasets:
- **MNIST**: A classic dataset of handwritten digits.
- **CIFAR-10**: A dataset of natural images spanning 10 categories.

### 4. 🚦 Metrics & Visualization
The defense's effectiveness is measured using metrics like:
- **AUC (Area Under Curve)**: Quantifies detection performance.
- **FPR (False Positive Rate)**: Measures false alarms during detection.
- **Detection Success Rate**: Indicates how effectively the defense differentiates between clean and adversarial inputs.

Below are examples of ROC curves showcasing detection performance:

![ROC Curve on MNIST](https://github.com/user-attachments/assets/84aa300c-e5c5-4d69-833b-98ba71677cdb)  
*ROC Curve of Detection on MNIST*

The above figure demonstrates how the detection mechanism performs on the MNIST dataset. The ROC curve highlights the trade-off between True Positive Rate (TPR) and False Positive Rate (FPR), with a high AUC indicating effective detection. This is particularly useful for simple, structured datasets like MNIST.

![ROC Curve on YouTube Face](https://github.com/user-attachments/assets/10dfd3e5-abfb-4397-991e-ee29d8de3b63)  
*ROC Curve of Detection on YouTube Face*

In contrast, the second figure shows the ROC curve for a more complex dataset: YouTube Faces. The defense mechanism is tested on real-world scenarios, demonstrating its robustness against adversarial attacks even in dynamic, unstructured environments like facial recognition.


---

## 📂 Project Structure

| File                | Description                                                                                        |
|---------------------|----------------------------------------------------------------------------------------------------|
| trap_utils.py      | Core utilities for model setup, dataset loading, and adversarial attack generation.                |
| inject_trapdoor.py | Script for injecting trapdoors into training datasets and training models with these patterns.      |
| eval_detection.py  | Script to evaluate model robustness and detect adversarial behaviors.                              |

---

## 🔧 Prerequisites

### Dependencies
Ensure the following are installed:
- 🐍 **Python 3.6+**
- 📚 **Keras**, **TensorFlow**, **NumPy**, **scikit-learn**

Install dependencies using:
```bash
pip install -r requirements.txt
```

## 🚀 Getting Started

### 1. Inject Trapdoors
To train a model with trapdoors, use the following command:
``` bash
python inject_trapdoor.py --gpu 0 --dataset mnist --inject-ratio 0.5
```
### 2. Evaluate Defense Mechanisms
To test the model against adversarial attacks, use:
``` bash
python eval_detection.py --gpu 0 --dataset cifar --attack all --filter-ratio 0.8
```
### 3. Command-Line Arguments

| Argument         | Description                                                |
|------------------|------------------------------------------------------------|
| --gpu          | GPU ID for training and evaluation.                        |
| --dataset      | Dataset to use (mnist or cifar).                        |
| --attack       | Type of attack (pgd, cw, en, or all).               |
| --inject-ratio | Ratio of adversarial examples in the training set.          |
| --filter-ratio | Proportion of neurons kept for detecting adversarial patterns. |
| --pattern-size | Size of trapdoor patterns injected into the dataset.        |

## 🧠 How It Works

| Step                  | Description                                                                               |
|-----------------------|-------------------------------------------------------------------------------------------|
| **Trapdoor Injection** | Generate trapdoor patterns and embed them into the dataset.                              |
| **Model Training**     | Train the model on both clean and adversarial datasets.                                  |
| **Adversarial Attacks**| Test the trained model against **PGD**, **CW**, and **ElasticNet** attacks.              |
| **Defense Evaluation** | Detect adversarial inputs using **neuron signatures** and compute metrics like **AUC**, **FPR**, and **Detection Success Rate**|   

![Cosine Similarity](https://github.com/user-attachments/assets/a00f261b-beb6-4fbc-8198-d4cdc86617ca)  
*Comparison of Cosine Similarity Between Normal and Adversarial Inputs*

In the figure above, the defense system's ability to distinguish between normal and adversarial inputs is demonstrated using cosine similarity. 
- **Trapdoored Models**: Show a clear separation between normal and adversarial inputs, with low similarity for adversarial inputs.
- **Non-Trapdoored Models**: Struggle to differentiate between the two, resulting in overlapping distributions.

This comparison highlights how trapdoor patterns enhance the model's ability to detect adversarial inputs effectively. By focusing on **neuron signatures**, the system achieves high accuracy in identifying adversarial behaviors.

## 🏁 Conclusion

The **AI Trapdoor Defense System** offers a cutting-edge approach to bolstering the security of deep learning models. By employing **trapdoor patterns** and **neuron signatures**, this system achieves:

- ✅ **Accurate Detection**: Identifies adversarial inputs with high precision.
- ⚡ **Minimal Performance Impact**: Maintains strong performance on clean data.
- 🛡️ **Resilience Against Attacks**: Demonstrates robust defense against diverse strategies, including **PGD**, **CW**, and **ElasticNet**.

This innovative method not only strengthens model robustness but also lays the groundwork for applying intentional adversarial patterns across various domains, including:
- 🖼️ **Image Recognition**
- 🎙️ **Speech Processing**
- 📚 **Natural Language Understanding**

---

## 💡 Looking Forward

The research community can expand on this work by:
- 🔍 **Exploring Adaptive Attacks**: Evaluate how resilient trapdoors are against dynamic adversarial strategies.
- 🌍 **Scaling to Real-World Data**: Test trapdoor mechanisms on larger, more complex datasets.
- 🔗 **Combining with Other Defenses**: Integrate trapdoor-based approaches with adversarial training techniques.

---

## 📧 Questions or Contributions?

We welcome collaboration and feedback from the community! Here’s how you can get involved:

1. 🤝 **Contribute**: Submit a pull request or open an issue in this repository.
2. 📩 **Get in Touch**: Reach out to us via
   - [ve086071@ucf.edu], [aa292153@ucf.edu], [ak748501@ucf.edu], [aj741009@ucf.edu]

Let’s work together to advance AI security and defend against adversarial threats!
