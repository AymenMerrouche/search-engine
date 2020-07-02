from collections import Counter
import numpy as np
from TextRepresenter import *
import re
from collections import OrderedDict
import copy

class Weighter():
    pass
    """
        La classe qui permet de pondérer un document.
    """
    def __init__(self, index):
        '''
        Constructor
        '''
        self.index = index
    def getWeightsForDoc(self,idDoc):
        pass
    def getWeightsForStem(self,stem):
        pass
    def getWeightsForQuery(self,query):
        pass
class FirstWeighter(Weighter):
    """
        Instanciation de la classe qui permet de pondérer un document.
    """
    def __init__(self, index):
        Weighter.__init__(self, index)
    # tf pour chaque terme du document
    def getWeightsForDoc(self, idDoc):
        return self.index.getTfsForDoc(idDoc)
    # tf dans chaque document ou il apparait
    def getWeightsForStem(self, stem):    
        return self.index.TfsForStem(stem)
    # 1 si le terme est dans la requete
    def getWeightsForQuery(self, query):
        proter_stemmer = PorterStemmer()
        # assurons nous que la requête ne contient que des termes du vocabulaire
        stems = list(set(self.index.indexInverse.keys()) & set(proter_stemmer.getTextRepresentation(query).keys()))
        tfs = {em : proter_stemmer.getTextRepresentation(query)[em] for em in stems}
        d = {term: 1 for term in tfs}
        return d


class SecondWeighter(Weighter):
    """
        Instanciation de la classe qui permet de pondérer un document.
    """
    def __init__(self, index):
        Weighter.__init__(self, index)

    # tf pour chaque terme du document
    def getWeightsForDoc(self, idDoc):
        return self.index.getTfsForDoc(idDoc)
    # tf dans chaque document ou il apparait   
    def getWeightsForStem(self, stem):
        return self.index.TfsForStem(stem)
    # tf de chaque terme de la requete dans la requete
    def getWeightsForQuery(self, query):
        proter_stemmer = PorterStemmer()
        # assurons nous que la requête ne contient que des termes du vocabulaire
        stems = list(set(self.index.indexInverse.keys()) & set(proter_stemmer.getTextRepresentation(query).keys()))
        tfs = {em : proter_stemmer.getTextRepresentation(query)[em] for em in stems}
        return tfs


class ThirdWeighter(Weighter):
    """
        Instanciation de la classe qui permet de pondérer un document.
    """
    def __init__(self, index):
        Weighter.__init__(self, index)
    
    # tf pour chaque terme du document
    def getWeightsForDoc(self, idDoc):
        return self.index.getTfsForDoc(idDoc)
    # tf dans chaque document ou il apparait
    def getWeightsForStem(self, stem):
        return self.index.TfsForStem(stem)
    # idf de chaque terme de la requete
    def getWeightsForQuery(self, query):
        proter_stemmer = PorterStemmer()
        # assurons nous que la requête ne contient que des termes du vocabulaire
        stems = list(set(self.index.indexInverse.keys()) & set(proter_stemmer.getTextRepresentation(query).keys()))
        tfs = {em : proter_stemmer.getTextRepresentation(query)[em] for em in stems}
        d = {term: self.index.getIDFForStem(term) for term in tfs}
        return d



class FourthWeighter(Weighter):
    """
        Instanciation de la classe qui permet de pondérer un document.
    """
    def __init__(self, index):
        Weighter.__init__(self, index)
    # 1 + log(tf) pour chaque terme du document
    def getWeightsForDoc(self, idDoc):
        tfs = self.index.getTfsForDoc(idDoc)
        return {term: 1 + np.log(tfs[term]) for term in tfs}
    # 1 + log(tf) dans chaque document ou il apparait
    def getWeightsForStem(self, stem):
        tfs = self.index.TfsForStem(stem)
        return {doc : 1 + np.log(tfs[doc]) for doc in tfs}
    # idf de chaque terme de la requete
    def getWeightsForQuery(self, query):
        proter_stemmer = PorterStemmer()
        # assurons nous que la requête ne contient que des termes du vocabulaire
        stems = list(set(self.index.indexInverse.keys()) & set(proter_stemmer.getTextRepresentation(query).keys()))
        tfs = {em : proter_stemmer.getTextRepresentation(query)[em] for em in stems}
        d = {term: self.index.getIDFForStem(term) for term in tfs}
        return d


class FifthWeighter(Weighter):
    """
        Instanciation de la classe qui permet de pondérer un document.
    """
    def __init__(self, index):
        Weighter.__init__(self, index)
    # 1 + log(tf) * idf pour chaque terme du document
    def getWeightsForDoc(self, idDoc):
        tfs = self.index.getTfsForDoc(idDoc)
        return {term: (1 + np.log(tfs[term]))*self.index.getIDFForStem(term) for term in tfs}
    # 1 + log(tf) * idf dans chaque document ou il apparait
    def getWeightsForStem(self, stem):
        tfs = self.index.TfsForStem(stem)
        return {doc : (1 + np.log(tfs[doc]))*self.index.getIDFForStem(stem) for doc in tfs}
    # 1 + log(tf) * idf de chaque terme de la requete
    def getWeightsForQuery(self, query):
        proter_stemmer = PorterStemmer()
        # assurons nous que la requête ne contient que des termes du vocabulaire
        stems = list(set(self.index.indexInverse.keys()) & set(proter_stemmer.getTextRepresentation(query).keys()))
        tfs = {em : proter_stemmer.getTextRepresentation(query)[em] for em in stems}
        d = {term: self.index.getIDFForStem(term)*(1 + np.log(tfs[term])) for term in tfs}
        return d

class IRModel():
    """
        Classe qui représente un modéle de RI
    """
    def __init__(self, index):
        '''
        Constructor
        '''
        self.index = index
    def getScores(self,query):
        """
        Renvoie les documents pertinents par rapport à la requête avec leur score
        """
        pass
    def getRanking(self,query):
        unsorted = self.getScores(query)
        sortedDict = sorted(unsorted.items(),reverse = True, key=lambda x: x[1])
        return sortedDict
class Vectoriel(IRModel):
    """
        Classe qui représente le modéle de RI Vectoriel
    """
    def __init__(self, index,weighter,normalized):
        '''
        Constructor
        '''
        IRModel.__init__(self, index)
        self.weighter = weighter 
        self.normalized = normalized

    def getScores(self,query):
        # X : representation de la query
        QueryTermWeights = self.weighter.getWeightsForQuery(query)
        # let's build our Y la representation du document

        # pour chaque stem de la query on retourne ses poids dans les documents dans lesquels il apparait
        stemWeights = {stem : self.weighter.getWeightsForStem(stem) for stem in QueryTermWeights}
        # on construit les document candidats, document qui est au moins rattaché à un stem de la query
        DocsCandidats = []
        for stem in stemWeights :
            DocsCandidats += stemWeights[stem].keys()
        DocsCandidats = list(set(DocsCandidats))

        # on genere les termes du cosine (X dot Y, Norme X , Norme Y)

        # X dot Y
        produitMatriciel = dict()
        # norme Y
        normeDocs = dict()

        # pour chaque document candidat
        for doc in DocsCandidats :
            # intialisation à 0import numpy as np

            produitMatriciel[doc] = 0
            # on parcours la query (le terme dois être dans le dico aussi)
            for stem in QueryTermWeights :
                # si ce terme apparait dans ce document autrement 0
                if doc in stemWeights[stem].keys():
                    # on ajoute un terme au prduit matriciel (X dot Y)
                    produitMatriciel[doc] += stemWeights[stem][doc]*QueryTermWeights[stem]
                    # on ajoute un terme à la norme de Y
                    if doc not in normeDocs.keys():
                        normeDocs[doc] = 0
                    normeDocs[doc] += stemWeights[stem][doc]**2 
        # la norme de Y
        normeQuery = np.linalg.norm(list(QueryTermWeights.values()))
        # formule de cosine
        cosine = {doc : produitMatriciel[doc] / (np.sqrt(normeDocs[doc])*normeQuery) for doc in produitMatriciel}
            
        if self.normalized:
            # score cosinus
            return cosine
        else:
            # produit scalaire
            return produitMatriciel
class ModeleLangue(IRModel):
    """
        Classe qui instancie le modéle de langue de RI
    """
    def __init__(self, index):
        '''
        Constructor
        '''
        IRModel.__init__(self, index)
    def getScores(self,query):
        # mise en forme de la query (separation, lemmisation...)
        proter_stemmer = PorterStemmer()
        # assurons nous que la requête ne contient que des termes du vocabulaire
        stems = list(set(self.index.indexInverse.keys()) & set(proter_stemmer.getTextRepresentation(query).keys()))
        tfs = {em : proter_stemmer.getTextRepresentation(query)[em] for em in stems}
        
        # lissage , requête courte
        if sum(tfs.values()) < 10 :
            lamda = 0.8
        # requête longue
        else:
            lamda = 0.2
            
        # Les documents candidats (contenants au moins une occurence d'un des termes de la requête)
        DocsCandidats = []
        for stem in tfs :
            DocsCandidats += self.index.indexInverse[stem].keys()
        DocsCandidats = list(set(DocsCandidats))
        
        # va contenir la probabilité de la query sachant le modéle et ce pour chaque document candidat
        probas = dict()
        
        
        # somme de tout les tfs
        s2 = sum([ sum(self.index.index[doc].values()) for doc in self.index.index])
        for doc in DocsCandidats:
            for stem in tfs:
                # si document pas encore rencontré, proba initialisé à 1 
                if doc not in probas.keys():
                    probas[doc] = 1
                    
                # probabilité de la requête dans le document
                probaDocument = 0
                # si le terme apparaît dans le document
                if doc in self.index.indexInverse[stem].keys():
                    # tf dans le dcument / somme des tfs des termes du document
                    probaDocument = self.index.indexInverse[stem][doc] / sum(self.index.index[doc].values())
                
                # probabilité de la requête dans le corpus entier
                probaCorpus = 0
                # somme des tf du terme
                s1 = sum(self.index.indexInverse[stem].values())
                # somme des tfs du terme / somme de tout les tfs
                probaCorpus = s1 / s2
                # formule du lissage Jelinek-Mercer
                probas[doc] = probas[doc]*((1-lamda)*probaDocument + lamda*probaCorpus)
        return probas
class Okapi(IRModel):
    """
        Classe qui instancie le modéle RI probabiliste BM25
    """
    def __init__(self, index, k1, b):
        '''
        Constructor
        '''
        IRModel.__init__(self, index)
        self.k1 = k1
        self.b = b
    def getScores(self,query):
        k1 = self.k1
        b = self.b
        # mise en forme de la query (separation, lemmisation...)
        proter_stemmer = PorterStemmer()
        # assurons nous que la requête ne contient que des termes du vocabulaire
        stems = list(set(self.index.indexInverse.keys()) & set(proter_stemmer.getTextRepresentation(query).keys()))
        tfs = {em : proter_stemmer.getTextRepresentation(query)[em] for em in stems}
            
        # Les documents candidats (contenants au moins une occurence d'un des termes de la requête)
        DocsCandidats = []
        for stem in tfs :
            DocsCandidats += self.index.indexInverse[stem].keys()
        DocsCandidats = list(set(DocsCandidats))
        
        # va contenir le score BM25
        scores = dict()
        # longueur moyenne d'un document
        avgdl = sum([ sum(self.index.index[doc].values()) for doc in self.index.index]) / len (self.index.index)
                                                                                        
        
        for doc in DocsCandidats:
            for stem in tfs:
                # si document pas encore rencontré, score inistialisé à 0
                if doc not in scores.keys():
                    scores[doc] = 0
                # si le terme apparaît dans la document
                if stem in self.index.index[doc].keys():
                    # idf du terme
                    idft = self.index.getIDFForStem(stem)
                    # tf du terme dans le document
                    tftd = self.index.index[doc][stem]
                    # longueur du document
                    D = sum(self.index.index[doc].values())
                    # on ecrit la formule
                    bm = (idft*tftd) / (tftd+k1*(1-b+b*(D/avgdl)))
                    scores[doc] += bm
        return scores

