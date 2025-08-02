import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.classify import SklearnClassifier
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from IPython.core.interactiveshell import InteractiveShell
import re
import os 
import random
import pandas as pd
import numpy as np
import csv
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore",category=DeprecationWarning)
from IPython.display import display


pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)


InteractiveShell.ast_node_interactivity="all"
#nltk.download('popular')
#nltk.download("averaged_perceptron_tagger_eng")
#nltk.download('tagsets')


sentance1="The big brown fox jumped over a lazy dog."
sentance2="This is particularly important in today's world where we are swamped with unstructured natural language data on the variety of social media platforms people engage in now-a-days (note- now-a-days in the decade of 2010-2020)"

#print(sentance1.lower())
#print(sentance2.lower())

def process_tokens(sentance):
    sentance=sentance.lower()
    tokenizer=RegexpTokenizer(r'\w+')
    tokens=tokenizer.tokenize(sentance) 
    filteredwords=[w for w in tokens if not w in stopwords.words("english")]
    return filteredwords


def extract_tagged(tags):
    tag_dict = nltk.data.load('help/tagsets/upenn_tagset.pickle')
    target_tags = list(tag_dict.keys())
    return [word for word, tag in tags if tag in target_tags]

def extract_features(sentance):
    processed_tokens=process_tokens(sentance)
    tags=nltk.pos_tag(processed_tokens)
    lmtzr=WordNetLemmatizer()
    sbs=SnowballStemmer("english")
    extracted_features=extract_tagged(tags)
    stemmed_words=[sbs.stem(w) for w in extracted_features]
    result=[lmtzr.lemmatize(w) for w in stemmed_words]
    return result
words=extract_features(sentance2)
#print(ans)
#print(extract_features("He hurt his right foot while he was wearing white shoes on his feet"))


def word_feats(words):
    return dict([(word,True) for word in words])
#print(word_feats(words))


def extract_features_from_doc(data):
    result=[]
    corpus=[]
    answers={}

    for (text,category,answer) in data:
        features=extract_features(text)
        corpus.append(features)
        result.append((word_feats(features),category))
        answers[category]=answer
    return (result,sum(corpus,[]),answers)
#print(extract_features_from_doc([['This is the input text from the user','category','answer to give']]))


def get_content(filename):
    doc=os.path.join(filename)
    with open(doc,'r') as content_file:
        lines=csv.reader(content_file,delimiter='|')
        data=[x for x in lines if len(x)==3]
        return data
data=get_content("leaves.txt")
features_data,corpus,answers=extract_features_from_doc(data)
#print(features_data[50])
#print(corpus)
#print(answer)

split_ratio=0.8

def split_dataset(data,split_ratio):
    random.shuffle(data)
    data_length=len(data)
    train_split=int(data_length*split_ratio)
    return (data[:train_split]),(data[train_split:])
training_data,test_data=split_dataset(features_data,split_ratio)
#print(traing_data)
#print(test_data)

np.save("training_data",training_data)
np.save("test_data",test_data)

training_data=np.load("training_data.npy",allow_pickle=True)
test_data=np.load("test_data.npy",allow_pickle=True)


def train_using_decision_tree(training_data,test_data):
    classifier=nltk.classify.DecisionTreeClassifier.train(training_data,entropy_cutoff=0.6,support_cutoff=6)
    classifier_name=type(classifier).__name__
    training_set_accuracy=nltk.classify.accuracy(classifier,training_data)
    #print("Trainnig Set Accuracy:",training_set_accuracy)
    test_set_accuracy=nltk.classify.accuracy(classifier,test_data)
    #print("Test Set Accuracy:",test_set_accuracy)
    return classifier,classifier_name,test_set_accuracy,training_set_accuracy
dtclassifier,classifier_name,test_set_accuracy,training_set_accuracy=train_using_decision_tree(training_data,test_data)


def train_using_naive_bayes(training_data,test_data):
    classifier=nltk.NaiveBayesClassifier.train(training_data)
    classifier_name=type(classifier).__name__
    training_set_accuracy=nltk.classify.accuracy(classifier,training_data)
    test_set_accuracy=nltk.classify.accuracy(classifier,test_data)
    return classifier,classifier_name,test_set_accuracy,training_set_accuracy
classifier,classifier_name,test_set_accuracy,training_set_accuracy=train_using_naive_bayes(training_data,test_data)
#print("Training set accuracy:",training_set_accuracy)
#print("Test set accuracy:",test_set_accuracy)
#print(len(classifier.most_informative_features()))
#classifier.show_most_informative_features()

#print(classifier.classify(({"leav":True,"use":True})))

#print(word_feats(extract_features("Hello")))

def reply(input_sentance):
    category=dtclassifier.classify(word_feats(extract_features(input_sentance)))
    return answers[category]

def input_output():
    while True:
        input_sentance=input("you: ")
        if input_sentance in ['exit','bye']:
            rply=reply(input_sentance)
            print(f"Bot: {rply}")
            break
        else:
            rply=reply(input_sentance) 
            print(f"Bot: {rply}")
            
if __name__=="__main__":
    input_output()