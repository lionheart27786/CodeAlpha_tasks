
# Image Recognition Task_1

This repository contains code for a deep learning model to classify rock, paper, and scissors hand gestures using TensorFlow and Keras. The model is trained on a dataset of hand gesture images, and data augmentation is employed to enhance the model's generalization.

## Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [License](#license)

## Introduction

This project focuses on building a convolutional neural network (CNN) to recognize rock, paper, and scissors hand gestures. The model is trained using a dataset consisting of labeled images, and data augmentation is applied to increase the diversity of the training set.

## Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/lionheart27786/CodeAlpha_tasks.git
   ```

2. Navigate to the project directory:

   ```bash
   cd rock-paper-scissors-classifier
   ```

3. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Download the training and test datasets:
   
   ```bash
   # Download the train set
   wget https://storage.googleapis.com/tensorflow-1-public/course2/week4/rps.zip
   
   # Download the test set
   wget https://storage.googleapis.com/tensorflow-1-public/course2/week4/rps-test-set.zip
   ```

2. Extract the datasets:

   ```bash
   unzip rps.zip -d tmp/rps-train
   unzip rps-test-set.zip -d tmp/rps-test
   ```

3. Run the Jupyter Notebook to explore the dataset and train the model:

   ```bash
   jupyter notebook Image_Recognition_Task_1.ipynb
   ```

## Data

The dataset consists of images representing hand gestures for rock, paper, and scissors. The training set is augmented using the `ImageDataGenerator` from TensorFlow to enhance the model's ability to generalize.

## Model Architecture

The neural network model is a sequential convolutional neural network (CNN) built using TensorFlow and Keras. It comprises convolutional layers, max-pooling layers, dropout for regularization, and dense layers for classification.

```python
# Include the code snippet of your model architecture here
```

## Training

The model is trained using the augmented training set. Training involves multiple epochs, and the model's performance is evaluated on a separate validation set. Data augmentation is employed to expose the model to a diverse range of hand gesture variations.

```python
# Include the code snippet for model training here
```

## Results

After training, the model's performance and accuracy can be observed by analyzing the training history and validation results. Evaluate the model on test data to assess its real-world performance.


