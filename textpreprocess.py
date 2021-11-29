import pandas as pd  
import numpy as np 
import re  
import nltk 

from nltk.corpus import stopwords
import multiprocessing as mp
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


class TextPreprocessor:
    def __init__(self, variety="BrE", user_abbrevs={}, n_jobs=1):
        """
        Text preprocessing transformer includes steps:
            1. Text normalization
            2. Punctuation removal
            3. Stop words removal
        """
        self.variety = variety
        self.user_abbrevs = user_abbrevs
        self.n_jobs = n_jobs

    def transform(self, X, *_):
        X_copy = X.copy()

        partitions = 1
        cores = mp.cpu_count()
        if self.n_jobs <= -1:
            partitions = cores
        elif self.n_jobs <= 0:
            return X_copy.apply(self._preprocess_text)
        else:
            partitions = min(self.n_jobs, cores)

        data_split = np.array_split(X_copy, partitions)
        pool = mp.Pool(cores)
        data = pd.concat(pool.map(self._preprocess_part, data_split))
        pool.close()
        pool.join()

        return data

    def _preprocess_part(self, part):
        return part.apply(self._preprocess_text)

    def _preprocess_text(self, message):
        # function for fully text preprocessing  like removing stop words, punctuations, lemmatization etc.
        message = message.lower()
        message = self._remove_punctuations(message)
        message = self._remove_stop_words(message)
        message = self._remove_urls(message)
        message = self._remove_numbers(message)
        message = self._stem_message(message)
        return message

    def _remove_punctuations(self, message):
        # function for removing punctuations
        import string

        message = message.translate(str.maketrans("", "", string.punctuation))
        return message

    def _remove_stop_words(self, message):
        from nltk.corpus import stopwords

        stop_words = set(stopwords.words("english"))
        words = message.split()
        words = [w for w in words if not w in stop_words]
        message = " ".join(words)
        return message

    def _remove_urls(self, message):
        message = re.sub(r"http\S+", "", message)
        return message

    def remove_hashtags(self, message):
        message = re.sub(r"#\S+", "", message)
        return message

    def _remove_numbers(self, message):
        message = re.sub(r"\d+", "", message)
        return message

    def _stem_message(self, message):
        # function for stemming message

        stemmer = PorterStemmer()
        message = " ".join([stemmer.stem(word) for word in message.split()])
        return message

    def _lemmatize_message(self, message):
        # function for lemmatizing message
        lemmatizer = WordNetLemmatizer()
        message = " ".join([lemmatizer.lemmatize(word) for word in message.split()])
        return message

    