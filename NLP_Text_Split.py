#import libraries
import nltk
import string
nltk.download_shell()
#Type: d -> stopwords -> q
from nltk.corpus import stopwords

#initial or read data
msgs= [line.rstrip() for line in open('SMSSpamCollection')]

#check format
for mess_no,msg in enumerate(msgs[:10]):
    print(mess_no,msg)
    print('\n')

#Create a Datagrame with each message as a tuple
import pandas as pd
msgs = pd.read_csv('SMSSpamCollection',sep='\t',names=['label','message'])
msgs['length'] = msgs['message'].apply(len) #length of each message
msgs['message'].head()

#Optional
#Visualization using Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#length distributions
msgs['length'].plot.hist(bins=100)
#spam vs normal messages
msgs.hist(column='length',by='label',bins=60)

#Processing
#Function to vectorize data
def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
#Example:
msgs['message'].head().apply(text_process)

#Using sklearn for CountVectorization
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(msgs['message'])

"""
#This does the folllowing:
mess4 = msgs['message'][3]
print(mess4)
bow4 = bow_transformer.transform([mess4])
#Generates a bag of words vector for number of times a word is repeated, ex: 4068 occurs twice
print(bow4)
#To get a words at index
print(bow_transformer.get_feature_names()[4068])
print(bow_transformer.get_feature_names()[9554])
"""

#Final Processing: Apply the example above on the complete dataset
msgs_bow = bow_transformer.transform(msgs['message'])
#This gives a sparse matrix with Rows: Messages & Cols: Words where the data is the frequency of a word in a message

#Calculating tfidf values
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_trans = TfidfTransformer().fit(msgs_bow)
"""
this converts the vector into a tfidf array of the message
tfidf4 = tfidf_trans.transform(bow4)
printf(tfidf4)
"""
#Calcualting tfidf of the whole dataset
msgs_tfidf = tfidf_trans.transform(msgs_bow)
#Now we have the whole dataset of sentences represented as numbers to which we can apply our models to predict certain outcomes