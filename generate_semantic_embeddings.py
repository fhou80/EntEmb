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
this file is for geneate fine-grained semantic entity embeddings, which
is the average of Word2Vec embeddings of type words.
"""

def load_type_vec(dic_dir, vec_dir):
    """
    dict is for indexing type words,
    type_vec is from google Word2Vec
    """
    type_vec = np.load(vec_dir)
    type2id = {}
    id2type = []
    with open(dic_dir, 'r') as fp:
        for line in fp:
            cont = line.split('\t')
            typ = cont[0]
            type2id[typ] = len(id2type)
            id2type.append(typ)
    return type2id, type_vec


def get_entity_emb(enti_file_dir, enti_file_list, type2id, type_vec, tee, saving_path):
    """
    generate semantic entity embeddings
    """
    tn, ed = type_vec.shape
    ent_indx = []
    ent_vecc = []
    num = 0
    for fi in enti_file_list:
        fid = enti_file_dir + fi
        with open(fid, 'r') as fp:
            for line in fp:
                num += 1
                if num % 1000 == 0:
                    print(num)
                cont = json.loads(line)
                e_name = cont[0]
                ot_list = cont[1]
                nt_list = []
                for ot in ot_list:
                    if ot not in type2id.keys():
                        bt = ot.split(' ')
                        for nt in bt:
                            if nt not in nt_list:
                                nt_list.append(nt)
                    else:
                        nt_list.append(ot)
                # use the nt_list
                #print(type_vec[0].shape)
                e_vec = np.zeros(ed)
                tn = 0
                for nnt in nt_list[:tee]:
                    if nnt in type2id.keys():
                        e_vec = e_vec + np.array(type_vec[type2id[nnt]])
                        tn += 1
                    else:
                        print('not in type Voc==='+nnt)
                type_num = np.ones(ed) * tn
                e_vec = e_vec / type_num
                #print(e_vec.shape)
                ent_indx.append(e_name)
                ent_vecc.append(e_vec)
    ent_vecc = np.array(ent_vecc)
    #print(ent_vecc.shape)
    print("Extracted {} entities".format(len(ent_indx)))
    with open(saving_path + '/dict_tee{}.entity'.format(tee), 'w') as ft:
        for ent in ent_indx:
            ft.write('en.wikipedia.org/wiki/' + ent+'\t11\n')
    np.save(saving_path + '/entity_vec_tee{}.npy'.format(tee), ent_vecc)


if __name__ == '__main__':
    type_dict = sys.argv[1]
    type_vec = sys.argv[2]    
    enti_file_dir = sys.argv[3]
    saving_path = sys.argv[4]
    tee = sys.argv[5]    
    type2id, type_vec = load_type_vec(type_dict, type_vec)    
    enti_file_list = ['single_entities.ndjson', 'picked_entities.ndjson']    
    get_entity_emb(enti_file_dir, enti_file_list, type2id, type_vec, tee, saving_path)
    