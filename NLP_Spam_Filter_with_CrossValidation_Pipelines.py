#We split the dataset into training and testing sets and apply ifidf and Naive Bayes to it
import nltk
import string
nltk.download_shell()
#Type: d -> stopwords -> q
from nltk.corpus import stopwords

#initial or read data
msgs= [line.rstrip() for line in open('SMSSpamCollection')]

#Create a Datagrame with each message as a tuple
import pandas as pd
msgs = pd.read_csv('SMSSpamCollection',sep='\t',names=['label','message'])
msgs['length'] = msgs['message'].apply(len) #length of each message

#Data Split
from sklearn.cross_validation import train_test_split
msg_train,msg_test,label_train,label_test = train_test_split(msgs['message'],msgs['label'],test_size=0.3)

#Import models and libraries for Vectorization and Classification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB        #I'm using Naive BAyes, but any Classifier can be used

#Using Pipelines to Vectorize training data and Classify it
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB())
])
#I'm using Naive BAyes, but any Classifier can be used

pipeline.fit(msg_train,label_train) #Train the model
pred = pipeline.predict(msg_test) #Predict labels

#To get the accuracy of the predictions, you can use Classification Reports:
from sklearn.metrics import classification_report
print(classification_report(label_test,pred)) #Gives about >95% accuracy