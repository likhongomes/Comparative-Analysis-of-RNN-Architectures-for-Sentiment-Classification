import pandas as pd
import numpy as np
import string
import nltk

nltk.download('treebank')
from nltk.tokenize import TreebankWordTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# import tensorflow as tf
# print(tf.__version__)
# import keras
# print(keras.__version__)

# from sklearn.model_selection import train_test_split

'''
Preprocess the text as follows:


Lowercase all text.
Remove punctuation and special characters.
Tokenize sentences (use Keras Tokenizer or nltk.word_tokenize).
Keep only the top 10,000 most frequent words.
Convert each review to a sequence of token IDs.
Pad or truncate sequences to fixed lengths of 25, 50, and 100 words (you will test these variations).

'''



def load_and_preprocess_data(file_path, max_len = 100):

    all_words = pd.DataFrame()

    def remove_sentence_breaks(sentence):
        return sentence.replace('<br />','')
    
    def leanify(sentence):
        for word in sentence:
            if word not in all_words['Word']:
                sentence.remove(word)
        return sentence
        # return [word for word in sentence if word in all_words['Word']]

    # Load data
    df = pd.read_csv(file_path)
    
    # Handle missing values
    df.dropna()

    ## Turn data to lowercase for all columns
    for col in df.columns:
        
        if col == 'review':
            ## Remove the sentence breaks <br />
            df[col] = df[col].apply(remove_sentence_breaks)
            ## Remove punctuation
            df[col] = df[col].str.lower()
            translator = str.maketrans('', '', string.punctuation)
            df[col] = df[col].str.translate(translator)
            ## Tokenize the reviews
            df[col] = df[col].apply(nltk.word_tokenize)
            

    all_words = df['review'].explode()
    word_frequencies = all_words.value_counts().reset_index()
    word_frequencies.columns = ['Word', 'Frequency']
    all_words = word_frequencies[0:10000]

    ## Leanify Sentence
    df['review'] = df['review'].apply(leanify)

    ## Convert to token ID
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['review'])
    df['review'] = tokenizer.texts_to_sequences(df['review'])
    
    ## Pad and Truncate the tokens
    df['review'] = list(pad_sequences(df["review"], maxlen=max_len, padding='post', truncating='post'))
            
    return df


if __name__ == "__main__":
    # Set pandas display options to show full content
    pd.set_option('display.max_colwidth', None)
    
    # Example usage
    df = load_and_preprocess_data('data/IMDB Dataset.csv',max_len = 100)

    print(len(set(df['review'].explode())))
    print(df.head())
