import argparse
import random
import sys

import numpy as np

from VIPER.perturbations_store import PerturbationsStorage

def ices(prob, text):
    raise NotImplementedError
    emb = "vce.normalized"
   
    results = []
    from gensim.models import KeyedVectors as W2Vec

    model = W2Vec.load_word2vec_format(emb)

    isOdd, isEven = False, False

    topn = 20

    mydict = {}

    # for line in texts:
    line = text
    a = line.split()
    wwords = []
    out_x = []
    for w in a:
        for c in w:
            if c not in mydict:
                similar = model.most_similar(c, topn=topn)
                if isOdd:
                    similar = [similar[iz] for iz in range(1, len(similar), 2)]
                elif isEven:
                    similar = [similar[iz] for iz in range(0, len(similar), 2)]
                words, probs = [x[0] for x in similar], np.array([x[1] for x in similar])
                probs /= np.sum(probs)
                mydict[c] = (words, probs)
            else:
                words, probs = mydict[c]
            r = random.random()
            if r < prob:
                s = np.random.choice(words, 1, replace=True, p=probs)[0]
                results.append(c, s)
            else:
                s = c
            out_x.append(s)
        # out_x.append(" ")
        wwords.append("".join(out_x))
        out_x = []

    print(" ".join(wwords))
    return results
    # perturbations_file.maybe_write()
