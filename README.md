# Spam Detection
-----------------

Spam detection is a supervised machine learning problem. This means you must provide your machine learning model with a set of examples of spam and ham messages and let it find the relevant patterns that separate the two different categories. Most email providers have their own vast data sets of labeled emails.


## 1. Loading Dependencies

We are going to make use of NLTK for processing the messages, WordCloud and matplotlib for visualization and pandas for loading data, NumPy for generating random probabilities for train-test split.

![Dependencies](https://github.com/whodoibenow/spamdetection/raw/main/Plots/Screenshot%202021-11-04%20at%207.39.42%20PM.png)

## 2. Train - Test Split

To test our model we should split the data into train dataset and test dataset. We shall use the train dataset t0 train the model and then it will be tested on the test dataset. We shall use 75% of the dataset as train dataset and the rest as test dataset. Selection of this 75% of the data is uniformly random.

![Spam vs Ham](https://github.com/whodoibenow/spamdetection/raw/main/Plots/Screenshot%202021-11-04%20at%207.40.23%20PM.png)

## 3. Length vs Spam

![](https://github.com/whodoibenow/spamdetection/raw/main/Plots/Unknow5n.png)

## 4. Visualisation of Data

![Code](https://github.com/whodoibenow/spamdetection/raw/main/Plots/Screenshot%202021-11-04%20at%207.50.01%20PM.png)

Let us see which are the most repeated words in the Ham messages! 

![](https://github.com/whodoibenow/spamdetection/raw/main/Plots/Unknown.png)

Let us see which are the most repeated words in the Ham messages! 


![](https://github.com/whodoibenow/spamdetection/raw/main/Plots/Unknown%20copy.png)

## 5. Training the model

We are going to implement two techniques: Bag of words and TF-IDF. I shall explain them one by one. Let us first start off with Bag of words.
Preprocessing: Before starting with training we must preprocess the messages. First of all, we shall make all the character lowercase. This is because ‘free’ and ‘FREE’ mean the same and we do not want to treat them as two different words.
Then we tokenize each message in the dataset. Tokenization is the task of splitting up a message into pieces and throwing away the punctuation characters. 

![](https://github.com/whodoibenow/spamdetection/blob/main/Plots/Screenshot%202021-11-04%20at%207.59.13%20PM.png)







