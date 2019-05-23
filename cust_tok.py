import nltk
from nltk.stem import WordNetLemmatizer
text = "hello it's can't be having"
text = text.lower().replace("'", "")
dict_tokens = {
    "its": "it is",
    "whats": "what is",
    "thats": "that is",
    "theres": "there is",
    "cant": "can not",
    "dont": "do not",
    "isnt": "is not",
    "wont": "will not",
    "doesnt": "does not",
    "hasnt": "has not",
    "didnt": "did not",
    "itd": "it would",
    "thatd": "that would",
    "whatd": "what would",
    "havent": "have not",
    "cannot": "can not",
    "couldnt": "could not",
}

for key, val in dict_tokens.items():
    text = text.replace(key, val)

wordnet_lemmatizer = WordNetLemmatizer()

words = nltk.word_tokenize(text)
words = [wordnet_lemmatizer.lemmatize(word, pos="v") for word in words]
print(words)

txt = ''
for w in words:
    txt += w+' '

tokens = []
for word in words:
    word_offset = txt.index(word, 0)
    word_len = len(word)
    running_offset = word_offset + word_len
    tokens.append(word_offset)
print(tokens)
