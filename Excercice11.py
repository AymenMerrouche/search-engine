from collections import Counter
import numpy as np
from TextRepresenter import *

def scoreVectTf(q,indexer):  
    porter_stemmer = PorterStemmer()
    normalizedReq = porter_stemmer.getTextRepresentation(q)
    indexInv = indexer.indexInverse
    scores=dict()
    for word in normalizedReq.keys():
        if word in indexInv.keys():
            for id,occ in indexInv[word].items():  
                if id in scores.keys():
                    scores[id]+=occ* normalizedReq[word]
                else:
                    scores[id] = occ *normalizedReq[word]
    return scores

# recuperer les termes uniques dans la requête
# recuperer le vecteur requete : comptage des termes dans la requête
# mesure de ressemblance entre la requête et le document (produit scalaire)

    