import json
import nltk
import spacy
import os
import re
from nltk.stem import WordNetLemmatizer
import sys
import numpy as np
import pickle
import operator
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

"""
this file is for geneate aggregated (hybrid) entity embeddings, which
is the linear aggregation of two sets of entity embeddings.
"""


def get_hybrid_embedding(dic1_dir, dic2_dir, emb1_dir, emb2_dir, alpha, saving_path):
    hybrid_ent_indx = []
    hybrid_ent_vecc = []
    tvec_1 = np.load(emb1_dir)
    tvec_2 = np.load(emb2_dir)
    enti2id_1 = {}
    id2enti_1 = []
    enti2id_2 = {}
    id2enti_2 = []
    with open(dic1_dir, 'r') as fp:
        for line in fp:
            cont = line.split('\t')
            ent = cont[0][22:]
            enti2id_1[ent] = len(id2enti_1)
            id2enti_1.append(ent)
    print(len(id2enti_1))
    with open(dic2_dir, 'r') as fp:
        for line in fp:
            cont = line.split('\t')
            ent = cont[0][22:]
            enti2id_2[ent] = len(id2enti_2)
            id2enti_2.append(ent)
    print(len(id2enti_2))
    for ent in id2enti_2:
        if ent in enti2id_1.keys():
            e_vec_1 = np.array(tvec_1[enti2id_1[ent]])
            e_vec_2 = np.array(tvec_2[enti2id_2[ent]])
            hybrid_vec = e_vec_1*alpha + e_vec_2*(1-alpha))
            hybrid_ent_indx.append(ent)
            hybrid_ent_vecc.append(hybrid_vec)
            #hybrid_ent_vecc.append(e_vec_1)
        else:
            hybrid_ent_indx.append(ent)
            hybrid_ent_vecc.append(np.array(tvec_2[enti2id_2[ent]]))
    hybrid_ent_vecc = np.array(hybrid_ent_vecc)
    with open(saving_path + '/aggregated_dict.entity', 'w') as ft:
        for typ in hybrid_ent_indx:
            ft.write('en.wikipedia.org/wiki/' + typ+'\t'+'11\n')
    np.save(saving_path + '/aggregated_entity_vec.npy', hybrid_ent_vecc)
    

if __name__ == '__main__':
    sem_emb_dict = sys.argv[1]
    sem_emb_vec = sys.argv[2]    
    ganea_emb_dict = sys.argv[3]
    ganea_emb_vec = sys.argv[4]
    saving_path = sys.argv[5]
    alpha = sys.argv[6]    
    get_hybrid_embedding(sem_emb_dict, ganea_emb_dict, sem_emb_vec, ganea_emb_vec, alpha, saving_path)