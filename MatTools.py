# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:06:01 2019

@author: mmorr
"""

import json
import pandas as pd
import spacy
from spacy import displacy
import en_core_web_sm
from pathlib import Path
import nltk
import heapq
import re
import unicodedata
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.toktok import ToktokTokenizer
from contractions import CONTRACTION_MAP
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize



tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')

file="C:\\Users\\mmorr\\Documents\\Jupyter_Notebook\\Deep Learning\\Deep Learning results.json"
filepath="C:\\Users\\mmor\\Documents\\Jupyter_Notebook\\Deep Learning\\Deep Learning results.csv"
with open(file) as f:
    df=json.load(f)
    db=pd.DataFrame(df)
    
def clean_content(content):
    
    text=content.replace('\n','').replace('_','').replace('<pre>','').replace('</pre>','').replace('\n','').replace('|','').strip()
    
    text=remove_accented_chars(text)
    
    text=expand_contractions(text)
    
#    text=remove_special_characters(text)
    
#    text=(lemmatize_text(text))
    
    return text
#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------

def build_summaries(df):
    
    summary_df=[]
    
    for title,doc in zip(df.title,df.text):

        print('processing '+str(title)+'...')

        article_text=clean_content(doc)

        formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )

        formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

        sentence_list = nltk.sent_tokenize(doc)

        stopwords = nltk.corpus.stopwords.words('english')

        word_frequencies = {}

        for word in nltk.word_tokenize(formatted_article_text):

            if word not in stopwords:

                if word not in word_frequencies.keys():

                    word_frequencies[word] = 1

                else:

                    word_frequencies[word] += 1

        try:
            maximum_frequncy = max(word_frequencies.values())
        except:
            summary_df.append({'title':title,'summary':'no_summary'})
            continue

        for word in word_frequencies.keys():

            word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

        sentence_scores = {}

        for sent in sentence_list:

            for word in nltk.word_tokenize(sent.lower()):

                if word in word_frequencies.keys():

                    if len(sent.split(' ')) < 30:

                        if sent not in sentence_scores.keys():

                            sentence_scores[sent] = word_frequencies[word]

                        else:

                            sentence_scores[sent] += word_frequencies[word]

        summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

        summary = ' '.join(summary_sentences)

        summary_df.append({'title':title,'summary':str(summary)})
        
    return summary_df
#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------

def build_labels(df):
    
    nlp = en_core_web_sm.load()
    
    labels_df=[]

    for title,doc in zip(df.title,df.text):

        article_text=clean_content(doc)
        
        doc=nlp(article_text)

        labels_df.append({'title':title,
                          
                          'Persons':[ent for ent in doc.ents if ent.label_=='PERSON'],
                           
                          'Institutions':[ent for ent in doc.ents if ent.label_=='GPE'],
                          
                          'Organizations':[ent for ent in doc.ents if ent.label_=='ORG']})
    return labels_df

#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
    
def remove_accented_chars(text):
    
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    return text

#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
    
def remove_special_characters(text, remove_digits=False):
    
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    
    text = re.sub(pattern, '', text)
    
    return text

#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------

def simple_stemmer(text):
    
    ps = nltk.porter.PorterStemmer()
    
    text = ' '.join([ps.stem(word) for word in text.split()])
    
    return text

#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------

def lemmatize_text(text):
    
    nlp = en_core_web_sm.load()

    text = nlp(text)
    
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    
    return text

#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
    
def build_summary(text):
    
    summary_df=[]
    
    article_text=clean_content(text)

    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )

    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    sentence_list = nltk.sent_tokenize(article_text)

    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}

    for word in nltk.word_tokenize(formatted_article_text):

        if word not in stopwords:

            if word not in word_frequencies.keys():

                word_frequencies[word] = 1

            else:

                word_frequencies[word] += 1


    maximum_frequncy = max(word_frequencies.values())
 
#    summary_df.append({'title':title,'summary':'no_summary'})
        

    for word in word_frequencies.keys():

        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

    sentence_scores = {}

    for sent in sentence_list:

        for word in nltk.word_tokenize(sent.lower()):

            if word in word_frequencies.keys():

                if len(sent.split(' ')) < 30:

                    if sent not in sentence_scores.keys():

                        sentence_scores[sent] = word_frequencies[word]

                    else:

                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)

#    summary_df.append({'title':title,'summary':str(summary)})
        
    return summary