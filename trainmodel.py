import nltk
import pandas as pd
from textpreprocess import TextPreprocessor
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
import pickle
import re
from sklearn.metrics import accuracy_score,fbeta_score,classification_report

nltk.download('stopwords')
stop=stopwords.words("english")

ss = SnowballStemmer("english")
ps = PorterStemmer()

msg_df = pd.read_csv('spam.csv', sep='\t', names=["label", "message"])

msg_df.groupby("message")["label"].agg([len, np.max]).sort_values(by = "len", ascending = False).head(n = 10)

msg_df["message"] = msg_df["message"].apply(cleanText)
msg_df.head(n = 10)

def encodeCategory(cat):
        if cat == "spam":
            return 1
        else:
            return 0

msg_df["label"] = msg_df["label"].apply(encodeCategory)

from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(encoding = "latin-1", strip_accents = "unicode")
features = vec.fit_transform(msg_df["message"])
print(features.shape)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X=cv.fit_transform(msg_df["message"])
print (X.shape)

X = cv.fit_transform(msg_df["message"]).toarray()
df = pd.DataFrame(X,columns=cv.get_feature_names())

y=msg_df['label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)

print(accuracy_score(y_test,y_pred))
print(fbeta_score(y_test,y_pred,beta =0.5))
print (classification_report(y_test,y_pred))

saved_model=pickle.dumps(spam_detect_model)
modelfrom_pickle = pickle.loads(saved_model) 
y_pred=modelfrom_pickle.predict(X_test)
print(accuracy_score(y_test,y_pred))

import joblib

joblib.dump(spam_detect_model,'pickle.pkl')

joblib.dump(X,'transform.pkl')