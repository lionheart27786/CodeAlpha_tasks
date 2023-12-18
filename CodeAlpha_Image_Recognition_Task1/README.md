# Rock Paper Scissors Classifier

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

```bash
# Clone the repository
git clone https://github.com/lionheart27786/CodeAlpha_tasks/tree/main/CodeAlpha_Image_Recognition_Task1
# Navigate to the project directory
cd rock-paper-scissors-classifier

# Install required Python packages
pip install -r requirements.txt
# Download the training and test datasets
wget https://storage.googleapis.com/tensorflow-1-public/course2/week4/rps.zip
wget https://storage.googleapis.com/tensorflow-1-public/course2/week4/rps-test-set.zip

# Extract the datasets
unzip rps.zip -d tmp/rps-train
unzip rps-test-set.zip -d tmp/rps-test

# Run the Jupyter Notebook to explore the dataset and train the model
jupyter notebook rock_paper_scissors_classifier.ipynb
