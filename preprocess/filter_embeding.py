# -*- coding: utf-8 -*-
import pickle
from tqdm import tqdm
import numpy as np
import sys
import nltk
import json
sys.path.append('../')

if __name__ == "__main__":
    embedding_weights = [ [0.0] * 300 ]
    token2id = {'[PAD]': 0 }
    id2token = ['[PAD]']

    for filedir in [
        'data/OTTers/train/', 'data/OTTers/dev/', 'data/OTTers/test/',
        'data/TGConv/train/', 'data/TGConv/dev/', 'data/TGConv/test/'
    ]:
        with open(filedir + 'concepts_nv.json') as f:
            lines = f.readlines()
            for row in tqdm(lines, desc="context words"):
                for sent in json.loads(row)['dialog']:
                    for token in nltk.word_tokenize(sent.lower()):
                        if token not in token2id:
                            id2token.append(token)
                            token2id[token] = len(id2token) - 1
                            embedding_weights.append([0.0] * 300)

    allgraphs = []
    allgraphs.extend(pickle.load(open('data/OTTers/train/graphs.pkl', 'rb')))
    allgraphs.extend(pickle.load(open('data/OTTers/dev/graphs.pkl', 'rb')))
    allgraphs.extend(pickle.load(open('data/OTTers/test/graphs.pkl', 'rb')))
    allgraphs.extend(pickle.load(open('data/TGConv/train/graphs.pkl', 'rb')))
    allgraphs.extend(pickle.load(open('data/TGConv/dev/graphs.pkl', 'rb')))
    allgraphs.extend(pickle.load(open('data/TGConv/test/graphs.pkl', 'rb')))

    for graph in tqdm(allgraphs, desc="graph nodes"):
        for node in list(graph.nodes):
            if node not in token2id:
                id2token.append(node)
                token2id[node] = len(id2token) - 1
                embedding_weights.append([0.0] * 300)

    missing_count = 0
    with open('raw_files/glove.6B.300d.txt') as f:
        total_len = 400001
        for line in tqdm(f, total = total_len, desc="filtering"):
            line = line.rstrip('\n').split(' ')
            if len(line) != 301:
                raise ValueError('vector error')
            word = line[0]
            if word not in token2id:
                missing_count += 0
                continue
            idx = token2id[word]
            embedding_weights[idx] = np.array(line[1:], dtype=np.float64)
    print('missing word ', missing_count)
    print('vocab size ', len(embedding_weights))
    with open('data/graph_embedding.pkl', 'wb') as f:
        pickle.dump((token2id, id2token, embedding_weights), f)

# python preprocess/5filter_embeding.py