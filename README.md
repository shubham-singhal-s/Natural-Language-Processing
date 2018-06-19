# Natural-Language-Processing
Python scripts for text vectorization and application of NLP to them for functions like Spam Filtering, Document Classification etc.
The dataset is not mine, so I cannot provide it, but here is the format:

**Natural Language Processing and Vectorization:**

ham\tNormal Text Message

ham\tNormal Text Message

spam\tSpam Message

# Introduction
This project is based on Sentiment Analysis that uses Machine Learning Classifiers to predict whethera given statement is of Positive or Negative Sentiment. The application of this project is to predict whether a user/consumer has a negative approach towards a product and thus to improve their experience. Another application is to monitor social media for negativity or hostility. The basic approach will be to use different classifiers to predict the sentiment and thus arrive at a concrete result.

# Data and Algorithm
**Data**
There are two parts of data:
1) Train/Test data
We use the movie reviews dataset provided in the nltk library to obtain two sets: Training and Testing. We use the training dataset to train our model and testing dataset to get the accuracy of each individual model we implement.
2) User Input
This will be the data that the user enters and wants to perform analysis on. This data is processed and an output is generated stating whether it's positive or negative.

**Algorithm**
The algorithm, written in Python 3.0 consists of a module and a python script. The script is used to train the models, and store them in pickle files.
