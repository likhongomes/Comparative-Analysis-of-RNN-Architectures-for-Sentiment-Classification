import pandas as pd
import numpy as np
import string
import nltk
nltk.download('treebank')
from nltk.tokenize import TreebankWordTokenizer

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

all_words = pd.DataFrame()

def load_and_preprocess_data(file_path):

    def remove_sentence_breaks(sentence):
        return sentence.replace('<br />','')
    
    def leanify(sentence):
        for word in sentence:
            if word not in all_words['Word']:
                sentence.remove(word)
        return sentence
    
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
    word_frequencies = word_frequencies[0:10000]

    for col in df.columns:
        ## Leanify Sentence
        df[col] = df[col].apply(leanify)
            
    return df



def word_frequencies(df):
    all_words = df['review'].explode()
    word_frequencies = all_words.value_counts().reset_index()
    word_frequencies.columns = ['Word', 'Frequency']
    word_frequencies = word_frequencies[0:10000]

    # print("\nWord frequencies across the entire DataFrame:")
    # print(word_frequencies)    

    return word_frequencies



# def retain_to_10k_words(sentence):
#     for word in sentence:
#         if word not in all_words['Word']:
#             sentence.remove(word)
#     return sentence

# def leanify_sentence(df):
#     df['review'] = df['review'].apply(retain_to_10k_words)
#     return df

if __name__ == "__main__":
    # Set pandas display options to show full content
    pd.set_option('display.max_colwidth', None)
    
    # Example usage
    processed_data = load_and_preprocess_data('data/IMDB Dataset.csv')
    # print(processed_data.head())

    all_words = word_frequencies(processed_data)
    print(all_words.shape)
    print(len(set(processed_data['review'].explode())))
    # leanify_sentence(processed_data)
    # # print(processed_data.head())
    # print(len(set(processed_data['review'].explode())))


    

# sentence = "One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic use of the word.<br /><br />It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.<br /><br />I would say the main appeal of the show is due to the fact that it goes where other shows wouldn't dare. Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side."
# print(sentence)
# print(">>>")
# def remove_sentence_breaks(sentence):
#         return sentence.replace('<br />','')

# print(remove_sentence_breaks(sentence))