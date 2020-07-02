from collections import Counter
import numpy as np
from TextRepresenter import *
import re
import copy

class Document():
    """
        La classe qui permet de stocker un document.
    """
    def __init__(self):
        '''
        Constructor
        '''
        self.I = ""
        self.T = ""
        self.B = ""
        self.A = ""
        self.K = ""
        self.W = ""
        self.X = ""
"""
    def __repr__(self):
        return self.I + " : " + self.T 
"""
class Parser():
    """
    permet de parser la collection stockée sous la forme d’un dictionnaire de Documents
    """
    def __init__(self):
        # le dictionnaire de documents
        self.documents = dict()
        
    def parsing(self,filename):
        # lecture du corpus et split sur .I
        corpus = open(filename).read().split(".I")
        del corpus[0]
        i=0
        while (i < len(corpus)):
            # extraction et structuration d'un document du corpus
            doc = corpus[i]
            d = Document()
            a = re.search(r"[0-9]+",doc)
            if a is not None:
                d.I = a.group(0)
                d.T = self.getElement("T",doc)
                d.B = self.getElement("B",doc)
                d.A = self.getElement("A",doc)
                d.K = self.getElement("K",doc)
                d.X = self.getElement("X",doc)  
                self.documents[str(d.I)] = d
                i+=1
            else:
                corpus[i-1]+=corpus[i]
                del corpus[i]
                i-=1
            

    def getElement(self,pattern,doc):
        res = re.search(r"\." + pattern + "([\s\S]*?)\.[ITBAKWX]",doc)
        if res is not None:
            return res.group(1)
        return ""

############ write the index and the inversed index in a file 

class IndexerSimple():
    """
        La classe qui permet d'indexer le corpus.
    """
    def __init__(self):
        self.index = dict()
        self.indexInverse = dict()
    
    def indexation(self,docs):

        porter_stemmer = PorterStemmer()
        for key,doc in docs.items():
            normalizedDoc = porter_stemmer.getTextRepresentation(doc.T)
            self.index[str(doc.I)] = normalizedDoc    
            for word,occurences in normalizedDoc.items():
                if word in self.indexInverse.keys():
                    if str(key) not in self.indexInverse[word]:
                        self.indexInverse[word][str(key)] = normalizedDoc[word]
                else:
                    self.indexInverse[word] = dict()
                    self.indexInverse[word][str(key)] = normalizedDoc[word]
            
        
    def getTfsForDoc(self,id):
        return self.index[str(id)]
    
    def getTfIDFsForDoc(self,id):
        N = len(self.index)
        doc = copy.deepcopy(self.index[str(id)])
        for k in self.index[str(id)].keys():
            doc[k] = self.index[str(id)][k] * np.log((1+N) /(1+ len(self.indexInverse[k])))
        return doc
            
    
    def TfsForStem(self,stem):
        if stem not in self.indexInverse.keys():
            return dict()
        return self.indexInverse[stem]

    def getIDFForStem(self,stem):
        N = len(self.index)
        if stem not in self.indexInverse.keys():
            return 0
        return np.log((1+N)/(1 + len(self.indexInverse[stem])))

    
    def getTfIDFsForStem(self,stem):
        if stem not in self.indexInverse.keys():
            return []
        N = len(self.index)
        stem_tfidf = copy.deepcopy(self.indexInverse[stem])
        for k in self.indexInverse[stem].keys():
            stem_tfidf[k] = self.index[str(k)][stem] * np.log((1+N)/(1 + len(self.indexInverse[stem])))
        return stem_tfidf
    
    def getStrDoc(self,parser,id):
        return parser.documents[str(id)].T