import nltk
import numpy as np
import pandas as pd
import regex as re
import unicodedata

from nltk.corpus import stopwords
from unicodedata import normalize


def basic_clean(string):
    '''
    Takes in a String and does basic text cleaning of lowercasing, normalising unicode (using NFKD, ascii, and utf-8), and removes any item that isn't a letter, number, whitespace, or single quote.
    Returns cleaned string.
    '''
    string = string.lower()
    string = normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8') 
    string = re.sub(r'[^a-z0-9\'\s]', '', string)
    return string


def tokenize(string):
    '''
    Takes in a String, tokenizes (seperates from punctuation) all the words in the string.
    Returns Tokenized String.
    '''
    token = nltk.tokenize.ToktokTokenizer()
    string = token.tokenize(string, return_str=True)
    return string

def full_clean(string, extra_words= [], exclude_words= [], language ='english'):
    string = tokenize(basic_clean(string))
    string = remove_stopwords(string,extra_words,exclude_words,language)
    return string


def stem(string):
    '''
    This function will accept a string and break it into individual text, then apply stemming (removes all but the root word [sometimes oddly]) to all the words, rejoin the text into a full string.
    Returns the newly stemmed string.
    [This is good for large file sizes, but accuracy is substantually lower, if you have less data to process or more processing power, consider using lemmatize() instead.]
    '''
    ps = nltk.PorterStemmer()
    stems = [ps.stem(word) for word in string.split(' ')]
    string = ' '.join(stems)
    return string


def lemmatize(string):
    '''
    FIRST TIME WILL REQUIRE INSTALL OF [ nltk.download('omw-1.4') ] AND [ nltk.download('wordnet') ] IT WILL GIVE YOU AN ERROR TELLING YOU TO INSTALL THEM OTHERWISE.
    This function will accept a string and break it into individual text, then apply lemmatization (replaces a word with it's dictionary form [usually]) to all the words, and rejoin the text into a full string.
    Returns the newly lemmatized string.
    [This is good for smaller file sizes, it has a higher readibility accuracy, but does require more computation power, if you have a lot of data or limited processing power consider stem() instead.]
    '''
    wnl = nltk.stem.WordNetLemmatizer()
    lemma = [wnl.lemmatize(word) for word in string.split(' ')]
    string= ' '.join(lemma)
    return string


def remove_stopwords(string, extra_words = [], exclude_words= [], language = 'english'):
    '''
    FIRST TIME REQUIRES THE INSTALLATION OF [ nltk.download('stopwords') ] IF NOT INSTALLED IT WILL GIVE YOU AN ERROR SAYING TO INSTALL IT.
    Takes in a string, as well as the option to add extra_words to the words that will be filtered out(stop words), and exclude_words to remove words from the filtered word list(stop words), Also added the ability to change the languge of the stop word list generated (default is english).
    Returns the string with stop words filtered out.
    '''
    stopwords_list = stopwords.words(language)
    if len(extra_words) > 0:
        stopwords_list.extend(extra_words)
    if len(exclude_words) > 0:
        set(stopwords_list) - set(exclude_words)
    words = string.split(' ')
    filtered_words = [word for word in words if word not in stopwords_list]
    string = ' '.join(filtered_words)
    return string
    