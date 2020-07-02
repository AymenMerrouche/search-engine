
import numpy as np
import collections
import re
import porter

"""
# Load library
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop_words = stopwords.words('english')
"""

#Global var to stock all the words of all documents
allwords = set()

def create_index_doc(nb):
    global allwords
    # Reading document and conversion to lower case
    words = re.findall(r'\w+', open('doc'+str(nb)+'.txt').read().lower())

    # Reading stop words
    stop_words = re.findall(r'\w+', open('stop_words.txt').read().lower())
    
    # Deleting stop words + normalization 
    words = [porter.stem(word) for word in words if word not in stop_words]

    #Saving all the words of all the documents
    allwords=set(list(allwords)+words)
    
    # Creation of dict
    return dict(collections.Counter(words))

def create_index(nb_docs):
    index=dict()
    # for each document 
    for i in range(nb_docs):
        index[i]=create_index_doc(i)
    return index

def create_index_inverse(nb_docs):
    index_inverse = dict()

    index = create_index(nb_docs)
    # for all the words
    for word in allwords:
        tmp=dict()
        # for each document 
        for doc in index :
            # if the document contains the word
            if word in (index[doc]).keys():
                # we add the document to the dictionary 
                tmp[doc]=index[doc][word]
        
        index_inverse[word]=tmp
    
    return index_inverse

def create_index_inverse_tf_idf(nb_docs):
    # Creating the index
    index = create_index(nb_docs)

    # Creating the inversed index
    index_inverse = create_index_inverse(nb_docs)


    index_inverse_tf_idf = dict()

    # for each word
    for word in allwords:
        tmp=dict()
        #for each document
        for doc in index :
            # if the document contains the word 
            if word in (index[doc]).keys():
                # Calculate the term frequency
                tf = index[doc][word]

                #Calculate the document frequency
                df = len(list(index_inverse[word]))

                idf = np.log((1+len(index))/(1+df))
                
                tmp[doc]=tf*idf
            else:    
                tmp[doc]=0
        
        index_inverse_tf_idf[word]=tmp
    
    return index_inverse_tf_idf

print(create_index_inverse_tf_idf(4))
