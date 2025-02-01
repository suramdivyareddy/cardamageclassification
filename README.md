# Car Damage Classification using Transfer Learning Models

## Overview

This project focuses on **classifying car damage** using deep learning models. It leverages **transfer learning** to efficiently detect and categorize damage levels in car images. By using pre-trained models, the project aims to automate the vehicle assessment process, which is crucial for industries such as insurance. This project uses **classification algorithms** and experiments with different transfer learning models, to find the most accurate one.

## Key Features

*   **Transfer Learning:** Uses MobileNet, MobileNetV2, and MobileNetV3 models for feature extraction.
*   **TensorFlow/Keras Implementation:** Utilizes fine-tuned TensorFlow/Keras models for classification.
*   **Image Preprocessing and Augmentation:** Implements image preprocessing and augmentation techniques to improve model performance.
*   **Colab Integration:** Supports Google Colab for easy execution and accessibility.
*   **Performance Visualization:** Provides visualizations for training and evaluation results.
*   **Simultaneous Training**: Trains two different transfer learning models simultaneously and identifies the model with higher accuracy.

## Problems Addressed

The increasing rate of car accidents has led to insurance companies facing significant challenges with claims processing. Existing systems for car damage detection often use masking algorithms and train models separately, which can be time-consuming and may not provide the most efficient results. This project aims to address these issues by:

*   **Automating the damage assessment process** using deep learning techniques.
*   **Improving efficiency** by using pre-trained models and transfer learning.
*   **Comparing performance** between different transfer learning models.

## Model Architecture

*   **Base Model**: MobileNet (pre-trained on ImageNet).
*   **Custom Layers**: Fully connected layers with ReLU activation and Dropout.
*   **Output Layer**: Single neuron with sigmoid activation (for binary classification) or can be extended for multiclass classification.

## Dataset and Preprocessing

*   Uses a dataset of car images labeled with damage severity.
*   Images are resized to 224x224 pixels.
*   Augmented with transformations for better generalization.

## Training and Evaluation

*   Trained using the Adam optimizer.
*   Uses ModelCheckpoint for selecting the best model.
*   Performance metrics include accuracy, loss plots, and a confusion matrix.

## Setup and Installation

1.  **Clone the Repository**
    ```
    git clone https://github.com/your-username/car-damage-classification.git
    cd car-damage-classification
    ```
2.  **Install Dependencies**
    ```
    pip install tensorflow numpy matplotlib seaborn pillow tqdm
    ```
3.  **Run the Notebook**

    Open `cardamage.ipynb` in Jupyter Notebook or Google Colab.

## Usage

1.  Upload a car image to the system.
2.  The model predicts whether the car is damaged or not, and can be extended to classify the damage level as minor, moderate or severe.

   ![Screenshot 2025-01-31 225500](https://github.com/user-attachments/assets/f94287e6-81f6-4b7d-9db7-959db16ea58b)


## Technologies Used

*   **Programming Language:** Python.
*  **Deep Learning Libraries**: TensorFlow, Keras.
*   **Convolutional Neural Networks**: CNN [5].
*   **Transfer Learning**: MobileNet, MobileNetV2, MobileNetV3.
*   **Tools**: Google Colab.

## Project Timeline

The project followed a structured timeline, as shown in the Gantt chart:

*   **Week 1:** Data Collection.
*   **Week 2-3:** Model Selection.
*   **Week 3-4:** Model Training.
*  **Week 4-5:** Model Testing.
*   **Week 5:** Output and Inferencing.
