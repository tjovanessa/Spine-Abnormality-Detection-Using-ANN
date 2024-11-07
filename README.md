# Spine-Abnormality-Detection-Using-ANN
This project focuses on automated spine abnormality detection, classifying individuals as "normal" or "abnormal" based on physical spine details using Artificial Neural Networks (ANN). It seeks to provide a tool for identifying potential lower back pain risks, helping healthcare professionals in early diagnosis and treatment planning.

### Overview
Lower back pain is a common condition that can arise from various spinal issues such as nerve irritation, muscle strain, disc degeneration, or damage to bones and ligaments. This project uses a dataset that includes physical spine details to help identify potential causes of lower back pain and classify individuals based on their risk levels.

By utilizing an ANN-based model, this project aims to improve the accuracy of diagnosing spine-related issues and enhance early intervention, supporting healthcare professionals in making informed decisions.

### Dataset
The dataset used for this project is the `dataset_spine.csv` dataset, available on Kaggle: <a href="https://www.kaggle.com/datasets/sammy123/lower-back-pain-symptoms-dataset">Spine Abnormality Detection Dataset</a>. It contains 310 observations along with 13 attributes (12 numeric predictors and 1 non-demographic binary class attribute) to predict whether a person is "Normal" or "Abnormal".

### Key Features
- Spine Abnormality Classification: Classifies individuals based on their spine-related data.
- ANN Model: Uses Artificial Neural Networks to analyze the data and predict outcomes.
- Early Diagnosis: Helps healthcare providers detect lower back pain risks and make better treatment decisions.

### About ANN
Artificial Neural Networks (ANNs) are a class of machine learning models inspired by the structure and function of the human brain. They are composed of layers of nodes (neurons) that process input data and learn from it through training.

**Basic Structure of an ANN:**
Input layer: Accepts the features from the dataset (in this case, physical spine data).
Hidden layers: Multiple layers of neurons that apply activation functions (e.g., Softmax, ReLU, Sigmoid) and learn patterns from the input data.
Output layer: Provides the final prediction (in this case, binary classification such as "Normal" vs "Abnormal")

**How ANN Works:**
- Forward Propagation: The input data is passed through the network, and each neuron in the layers computes a weighted sum of the inputs and applies an activation function to produce an output.
- Loss Calculation: The model's predictions are compared to the actual values (true labels), and a loss function measures the error.
- Backpropagation: The error is propagated backward through the network, adjusting the weights of the neurons to minimize the loss.
- Optimization: This process is repeated iteratively using optimization techniques (like Gradient Descent) to improve the model's accuracy over time.

ANNs are particularly effective for pattern recognition tasks like image recognition, speech processing, and, as in this case, classification tasks based on structured data.

### Getting Started
1. Clone this repository to your local.
2. Download the `dataset_spine.csv` dataset from this repository and load it.
3. Run the spine_detection.py script to train and test the model, results including loss plot and model evaluation metrics will be displayed.

### Requirements
To run this project, ensure you have the following dependencies installed:

`Python 3.x` <br>
`pandas` <br>
`numpy` <br>
`scikit-learn` <br>
`tensorflow` (for ANN model) <br>
`matplotlib` (for visualization)
