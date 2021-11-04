# Spam Detection
-----------------

Spam detection is a supervised machine learning problem. This means you must provide your machine learning model with a set of examples of spam and ham messages and let it find the relevant patterns that separate the two different categories. Most email providers have their own vast data sets of labeled emails.


## 1. Loading Dependencies

We are going to make use of NLTK for processing the messages, WordCloud and matplotlib for visualization and pandas for loading data, NumPy for generating random probabilities for train-test split.

![Dependencies](https://github.com/whodoibenow/spamdetection/raw/main/Plots/Screenshot%202021-11-04%20at%207.39.42%20PM.png)

## 2. Train - Test Split

To test our model we should split the data into train dataset and test dataset. We shall use the train dataset t0 train the model and then it will be tested on the test dataset. We shall use 75% of the dataset as train dataset and the rest as test dataset. Selection of this 75% of the data is uniformly random.

![Spam vs Ham](



