import pandas as pd
df=pd.read_csv("sentiment_data.csv")
#performing EDA
print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.isnull().sum())
print(df["sentiment"].value_counts())
#encoding the positive and negative as 0 and 1 so model can understand easily

df["sentiment"]=df["sentiment"].map({"positive":1, "negative":0})
print(df.head())

#cleaning the data for the model
df["review"]=df["review"].apply(lambda x : x.lower())
import string
def remove_punc(txt):
    return txt.translate(str.maketrans('','',string.punctuation))
df["review"]=df["review"].apply(remove_punc)

def remove_numbers(txt):
    new=""
    for i in txt:
        if not i.isdigit():
            new+=i
    return new

df["review"]=df["review"].apply(remove_numbers)

def remove_emojis(txt):
    new=""
    for i in txt:
        if i.isascii():
            new+=i
    return new
df["review"]=df["review"].apply(remove_emojis)

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

stopwords=set(stopwords.words('english'))

def remove_stopwords(txt):
    words=txt.split()
    cleaned=[]
    for i in words:
        if i not in stopwords:
            cleaned.append(i)
    return " ".join(cleaned)
df["review"]=df["review"].apply(remove_stopwords)

def remove_html_tags(txt):
    new=""
    inside_tag=False
    for i in txt:
        if i=="<":
            inside_tag=True
        elif i==">":
            inside_tag=False
        elif not inside_tag:
            new+=i
    return new
df["review"]=df["review"].apply(remove_html_tags)
print(df.head())

# data cleaning is done and now going to training and testing

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#here we are importing only accuracy because the data is balanced. we have already checked the class distribution

y=df["sentiment"]
x_train,x_test,y_train,y_test=train_test_split(df["review"],y,test_size=0.2,random_state=42,stratify=y)

bow_vectorizer=CountVectorizer()
bow_train=bow_vectorizer.fit_transform(x_train)
bow_test=bow_vectorizer.transform(x_test)

tf_vectorizer=TfidfVectorizer()
tf_train=tf_vectorizer.fit_transform(x_train)
tf_test=tf_vectorizer.transform(x_test)

nb_model=MultinomialNB()
nb_model.fit(bow_train,y_train)
bow_prediction=nb_model.predict(bow_test)

nb2_model=MultinomialNB()
nb2_model.fit(tf_train,y_train)
tf_prediction=nb2_model.predict(tf_test)

print("Accuracy score using naive bayes for bag of words model was:",accuracy_score(y_test,bow_prediction))
print("Accuracy score using naive bayes for tfidf model was :",accuracy_score(y_test,tf_prediction))

log_bow_model=LogisticRegression(max_iter=1000)
log_bow_model.fit(bow_train,y_train)
log_bow_prediction=log_bow_model.predict(bow_test)

log_tf_model=LogisticRegression(max_iter=1000)
log_tf_model.fit(tf_train,y_train)
log_tf_prediction=log_tf_model.predict(tf_test)

print("Accuracy score using Logistic Regression for tfidf model was :",accuracy_score(y_test,log_tf_prediction))

print("Accuracy score using Logistic Regression for bag of words model was: ",accuracy_score(y_test,log_bow_prediction))


#best model was logistic regression tfidf model
print("best model classification report and confusion matrix")
print("classification Report :",classification_report(y_test,log_tf_prediction))
print("confusion Matrix:",confusion_matrix(y_test,log_tf_prediction))
review=input("Enter the review of the movie as you like ")
review_check=tf_vectorizer.transform([review])
prediction=log_tf_model.predict(review_check)
if prediction[0]==1:
    print("positive")
else:
    print("negative")