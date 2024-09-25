
# Spam vs Ham SMS Classification

This project demonstrates a complete end-to-end process for classifying SMS messages as either spam or ham (not spam). It involves data preprocessing, visualization, undersampling, tokenization, and training an LSTM model using TensorFlow/Keras for binary classification.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [License](#license)

## Project Overview

This project performs a spam/ham classification using a dataset of SMS messages. The dataset is first cleaned and preprocessed, followed by visualization. An LSTM-based deep learning model is then trained to distinguish between spam and ham messages.

## Dataset

The dataset used is the SMS Spam Collection Dataset. It consists of two columns:
- `v1`: The label of the message (ham or spam)
- `v2`: The actual message

You can download the dataset from [here](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/spam-vs-ham-classification.git
    cd spam-vs-ham-classification
    ```

2. Install the required libraries:
    ```bash
    pip install numpy pandas seaborn matplotlib tensorflow scikit-learn nltk
    ```

3. Download the NLTK stopwords by running:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

## Exploratory Data Analysis

The dataset undergoes an initial exploratory data analysis:
- Displaying the first few rows
- Checking for missing values
- Visualizing message length distribution
- Comparing the length of spam vs ham messages

The project uses Seaborn and Matplotlib to generate informative visualizations such as histograms and count plots.

## Preprocessing

Preprocessing involves:
- Dropping unnecessary columns
- Renaming columns for clarity
- Calculating message lengths
- Balancing the dataset via undersampling
- Text cleaning and tokenization using NLTK (e.g., stopword removal, stemming)

The data is then transformed into sequences using `keras.preprocessing.text.one_hot` and padded to ensure uniform length.

## Model Architecture

The project uses an LSTM-based neural network model built using TensorFlow/Keras. The architecture includes:
- Embedding Layer
- LSTM Layer
- Dense Output Layer

### Model Parameters

- Vocabulary Size: 10,000
- Sequence Length: 200
- LSTM Units: 128
- Optimizer: Adam with a learning rate of 0.001
- Loss Function: Binary Cross-Entropy

## Training

The model is trained on the preprocessed data using the `model.fit()` function with the following parameters:
- Training for 10 epochs
- Using 15% of the data for validation

The model achieved the following training/validation accuracy:
- Training Accuracy: ~99.8%
- Validation Accuracy: ~97.9%

## Results

The final model demonstrates high accuracy in distinguishing between spam and ham messages. A sample output of training accuracy is as follows:

| Epoch | Training Accuracy | Validation Accuracy |
|-------|--------------------|---------------------|
| 1     | 79.22%             | 93.19%              |
| 10    | 99.81%             | 97.91%              |

## License

This project is licensed under the MIT License.
