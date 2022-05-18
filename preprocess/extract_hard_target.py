import json
import pickle

token2id, id2token, weights = pickle.load(open('data/graph_embedding.pkl', 'rb'))

hard_target = []
with open('data/word_frequency_list_60000_English.txt', 'r') as f:
    for row in f:
        data = row.rstrip('\n').split(',')
        if data[0] == 'N' and int(data[2]) < 800 and int(data[2]) > 200 and data[1] in token2id:
            hard_target.append(data[1])

with open('data/hard_target.json', 'w') as f:
    json.dump(hard_target, f)

# python preprocess/extract_hard_target.py