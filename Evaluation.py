from collections import Counter
import numpy as np
from TextRepresenter import *
import re
import copy
import math
from scipy import stats
class Query():
    """
        La classe qui permet de stocker une query.
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
        # on y met les documents pertinents de cette requête
        self.per = []
"""
    def __repr__(self):
        return self.I + " : " + self.T 
"""
class QueryParser():
    """
    permet de parser la collection de query stockée sous la forme d’un dictionnaire de Query
    """
    def __init__(self):
        # le dictionnaire des requêtes
        self.queries = dict()
        
    def parsing(self,filenameQ, filenameJ):
        # lecture du corpus et split sur .I
        corpus = open(filenameQ).read().split(".I")
        del corpus[0]
        i=0
        while (i < len(corpus)):
            # extraction et structuration d'une query du corpus
            qry = corpus[i]
            q = Query()
            a = re.search(r"[0-9]+",qry)
            if a is not None:
                q.I = a.group(0)
                q.T = self.getElement("T",qry+".I")
                q.B = self.getElement("B",qry+".I")
                q.A = self.getElement("A",qry+".I")
                q.K = self.getElement("K",qry+".I")
                q.W = self.getElement("W",qry+".I")
                q.X = self.getElement("X",qry+".I")  
                self.queries[str(q.I)] = q
                i+=1
            else:
                corpus[i-1]+=corpus[i]
                del corpus[i]
                i-=1
                
                
        # lecture de listes de pertinences et split sur \n
        relevances = open(filenameJ).read().split("\n")
        i=0
        while (i < len(relevances)):
            # extraction et construction de la liste de documents pertinents d'une requête
            rel = relevances[i]
            a = re.search(r"[0-9]+",rel)
            if a is not None:
                ide = a.group(0)
                b = re.search(r"[0-9]+",rel[a.end():])
                p = b.group(0)
                self.queries[str(ide)].per.append(str(p))
            i+=1
    def getElement(self,pattern,qry):
        res = re.search(r"\." + pattern + "([\s\S]*?)\.[ITBAKWX]",qry)
        if res is not None:
            return res.group(1)
        return ""
    
class EvalMesure():
    """
        La classe qui permet d'évaluer le rendu d'un modéle de RI.
    """
    def __init__(self):
        pass
    def evalQuery(self,liste,query):
        pass
                                                
class Rapel(EvalMesure):
    """
        La classe qui permet d'évaluer le rendu d'un modéle de RI selon la mesure de rappel.
    """
    def __init__(self,k):
        EvalMesure.__init__(self)
        self.k = k
    def evalQuery(self,liste,query):
        # doc pertinents
        pertinets = query.per
        # doc pertinents et renvoyés
        pertinetsRetournes = list(set(liste[:self.k]) & set(pertinets))
        # formule du rapel
        if len(pertinets) == 0:
            return 1
        return len(pertinetsRetournes) / len(pertinets)

class Precision(EvalMesure):
    """
        La classe qui permet d'évaluer le rendu d'un modéle de RI selon la mesure de precision.
    """
    def __init__(self,k):
        EvalMesure.__init__(self)
        self.k = k
    def evalQuery(self,liste,query):
        # docs pertinents
        pertinets = query.per
        # docs pertinents et renvoyés
        pertinetsRetournes = list(set(liste[:self.k]) & set(pertinets))
        # formule de la precision
        return len(pertinetsRetournes) / self.k
    
class Fmesure(EvalMesure):
    """
        La classe qui permet d'évaluer le rendu d'un modéle de RI selon la Fmesure.
    """
    def __init__(self,k,beta):
        EvalMesure.__init__(self)
        self.k = k
        self.beta = beta
    def evalQuery(self,liste,query):
        # on calcule la precision et le rappel au rang k
        pre = Precision(self.k).evalQuery(liste,query)
        rap = Rapel(self.k).evalQuery(liste,query)
        # formule de la Fmesure
        if pre == 0:
            return 0
        return (1+self.beta**2)*((pre*rap) / ((self.beta**2)*pre + rap))
    
class PrecisionMoyenne(EvalMesure):
    """
        La classe qui permet d'évaluer le rendu d'un modéle de RI selon la precisionMoyenne.
    """
    def __init__(self):
        EvalMesure.__init__(self)
    def evalQuery(self,liste,query):
        res = 0
        k=1
        # pour chaque document renvoyé
        for d in liste:
            # s'il est pertinent
            if d in query.per:
                # on met à jour la somme
                res +=Precision(k).evalQuery(liste,query)
            k+=1
        if len(list(set(liste) & set(query.per))) == 0:
            return 0
        return res/len(list(set(liste) & set(query.per)))
    
class ReciprocalRank(EvalMesure):
    """
        La classe qui permet d'évaluer le rendu d'un modéle de RI selon le ReciprocalRank.
    """
    def __init__(self):
        EvalMesure.__init__(self)
    def evalQuery(self,liste,query):
        i = 1
        trgt = False
        # pour chaque doc retourné
        for d in liste :
            # le document pertinent de plus haut rang
            if d in query.per:
                trgt = True
                break
            i += 1
        if trgt == False :
            return 0
        return 1/i

class ndcg(EvalMesure):
    """
        La classe qui permet d'évaluer le rendu d'un modéle de RI selon le nDCGp.
    """
    def __init__(self,p):
        EvalMesure.__init__(self)
        self.p = p
    def evalQuery(self,liste,query):
        # si pertinent rel1 = 1 sinon 0
        dcgp = 0
        if len(liste)>0 : 
            if liste[0] in query.per:
                dcgp = 1
        for i in range(1,min(self.p,len(liste))):
            # si pertinent reli = 1 sinon 0
            if liste[i] in query.per:
                dcgp += 1/math.log2(i+1)
        dcgpi = 1
        for i in range(1,self.p):
            dcgpi += 1/math.log2(i+1)
        return dcgp / dcgpi               
class EvalIRModel:
    """
        La classe qui permet d'évaluer le rendu d'un modéle de RI selon une mesure et un ensemble de test passés en param.
    """
    def __init__(self):
        pass
    
    def evaluate(self,modelIR, mesure, queries):
        # On recupere le ranking
        values = []
        for q in queries :
           
            ranking = [[*x] for x in zip(*modelIR.getRanking(queries[q].W))]
            # Si on trouve des documents pertinets
            if len(ranking) > 0:
                ranking = ranking[0]
            values.append(mesure.evalQuery(ranking,queries[q]))
        mean = sum(values) / len(values) 
        variance = sum([((x - mean) ** 2) for x in values]) / len(values) 
        sigma = variance ** 0.5
        return [mean, sigma]
    def test_teta(self, srocre1, score2, seuil = 0.05):
        a, b = stats.ttest_ind
        if b < seuil:
            print("performances differentes au risque 0.05", a)
        else :
            print("performances ne sont pas differentes au risque 0.05", a)