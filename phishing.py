from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.stem.snowball import SnowballStemmer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd
import joblib
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv("phishing_site_urls.csv")
df.head()
df.isnull().sum()
df.describe()
df = df.drop_duplicates()
tokenizer = RegexpTokenizer(r'[A-Za-z]+')
df['text_tokenized'] = df.URL.map(lambda t: tokenizer.tokenize(t))
df.head()
root_words = SnowballStemmer("english")
df['root_words'] = df['text_tokenized'].map(
    lambda l: [root_words.stem(word) for word in l])
df.head()
df['text_sent'] = df['root_words'].map(lambda l: ' '.join(l))
df.head()
bad_sites = df[df.Label == 'bad']
good_sites = df[df.Label == 'good']
bad_sites.head()
good_sites.head()
print(list(STOPWORDS)[:10])
c = CountVectorizer()
cv = c.fit_transform(df.text_sent)
print(list(c.vocabulary_)[:10])
print('The length of vocabulary', len(c.get_feature_names()))
print('The shape is', cv.shape)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    cv, df.Label, test_size=0.3, random_state=42)
lr = LogisticRegression(max_iter=507197)
lr.fit(Xtrain, Ytrain)
lr.score(Xtest, Ytest)
ypred = lr.predict(Xtest)
print('\nCLASSIFICATION REPORT\n')
print(classification_report(ypred, Ytest, target_names=['Bad', 'Good']))
Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    df.URL, df.Label, test_size=0.3, random_state=42)
pipeline_ls = make_pipeline(CountVectorizer(tokenizer=RegexpTokenizer(
    r'[A-Za-z]+').tokenize, stop_words='english'), LogisticRegression(max_iter=507197))
pipeline_ls.fit(Xtrain, Ytrain)
phish = []

n = int(input("Enter number of elements : "))

for i in range(0, n):
    ele = [input()]
    phish.append(ele)
    result1 = pipeline_ls.predict(ele)
    print(result1)

