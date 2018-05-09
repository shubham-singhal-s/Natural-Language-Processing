#Based on our previous project, where we extracted tdidf values from the messages, we apply Naive Bayes (or any other Classification) to get the labels: 'ham' or 'spam'

from sklearn.naive_bayes import MultinomialNB
#Naive Bayes Classifier for Multinomial Dist

spam_detector = MultinomialNB().fit(msgs_tfidf,msgs['label'])
#Correlates ham and spam labels to tfidf values

all_pred = spam_detector.predict(msgs_tfidf)
#gives an array of ham or spam values based on tfidf vlues

#example:
allpred[4]
#Classifies the message as 'ham', let's verify:
msgs['label'] #Gives 'ham'
msgs['message'] #A normal text message

#Thus we have trained a model to test for Spam

#For tsting the model, refer cross validation files