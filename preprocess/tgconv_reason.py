# -*- coding: utf-8 -*-
import pickle
import json
import networkx as nx
from tqdm import tqdm
from functools import partial
import multiprocessing
import sys
from nltk.stem import PorterStemmer
import random
import numpy as np
from collections import Counter
sys.path.append('../')

def handle_func(ns, start_idx):
    print("handle for every conversation...")
    batch_data = ns.all_data[start_idx: start_idx + ns.batch_size]
    concept_net = ns.concept_net
    concept2id = ns.concept2id
    pst = PorterStemmer() 
    with open('data/hard_target.json', encoding='utf-8') as f:
        hard_target_set = json.load(f)

    def add_edges(source, target, sub_graph: nx.Graph):
        ans = []
        for s in source:
            for t in target:
                if s not in concept2id or t not in concept2id:
                    continue
                sub_graph.add_node(s, x=s)
                sub_graph.add_node(t, x=t)
                if s == t:
                    continue
                res = list(nx.all_simple_edge_paths(concept_net, source=s, target=t, cutoff=2))
                ans.extend(res)
                for path in res:
                    for pair in path:
                        p0 = pst.stem(pair[0])
                        p1 = pst.stem(pair[1])
                        if p0 in concept2id and p1 in concept2id:
                            sub_graph.add_node(p0, x=p0)
                            sub_graph.add_node(p1, x=p1)
                            sub_graph.add_edge(p0, p1)

        tempnodes = source.copy()
        tempnodes.extend(target)
        for c in tempnodes:
            nei1 = list(set([pst.stem(n1) for n1 in list(concept_net.neighbors(c)) ]))
            random.shuffle(nei1)
            nei1 = [ n1 for n1 in nei1 if n1 in concept2id ][:5]
            for n1 in nei1:
                sub_graph.add_node(n1, x=n1)
                sub_graph.add_edge(c, n1)

                nei2 = list(set([pst.stem(n2) for n2 in list(concept_net.neighbors(n1)) ]))
                random.shuffle(nei2)
                nei2 = [ n2 for n2 in nei2 if n2 in concept2id ][:5]
                for n2 in nei2:
                    sub_graph.add_node(n2, x=n2)
                    sub_graph.add_edge(n1, n2)
        return ans

    batch_result = []
    connect_path_counter = []
    for data in tqdm(batch_data, mininterval=30):
        dialog = data['session'].copy()
        concepts = data['concepts'].copy()
        len_c = len(concepts)
        continue_next_sign = [ 0 for i in range(len_c - 1) ]
        continue_path = []
        continue_graphs = []
        for i in range(len_c - 1):
            sub_graph = nx.Graph()
            res = add_edges(concepts[i], concepts[i+1], sub_graph)
            connect_path_counter.append(len(res))
            continue_graphs.append(sub_graph)
            continue_next_sign[i] = str(i) if len(res) > 10 else '-'
            start = Counter([path[0][0] for path in res]).most_common(1)
            continue_path.append('-' if len(start) == 0 else start[0][0])

        end = Counter([path[-1][-1] for path in res]).most_common(1)
        continue_path.append('-' if len(end) == 0 else end[0][0])

        max_len = [0, 0]
        for seq in ','.join(continue_next_sign).split('-'):
            num_seq = [ int(i) for i in seq.split(',') if i != '' ]
            if len(num_seq) == 0:
                continue
            start_idx = min(num_seq)
            end_idx = max(num_seq)
            if end_idx - start_idx >= max_len[1] - max_len[0]:
                max_len = [start_idx, end_idx]

        dialog = dialog[max_len[0]:max_len[1] + 2]
        concepts = concepts[max_len[0]:max_len[1] + 2]
        entity_path = continue_path[max_len[0]:max_len[1] + 2]
        easy_target = entity_path[-1]
        if easy_target == '-':
            continue
        if len(dialog) <= 3:
            continue
        global_graph = nx.Graph()
        for sub_graph in continue_graphs[max_len[0]:max_len[1] + 2]:
            global_graph = nx.compose(global_graph, sub_graph) 

        hard_target = random.choice(hard_target_set)

        batch_result.append((
            dialog, concepts, global_graph, easy_target, hard_target, entity_path
        ))
    print('connect_path_counter ', np.median(connect_path_counter))
    return batch_result


def main(filedir):
    with open(filedir + 'raw.json') as f:
        alldata = [json.loads(row) for row in f]
    
    concept_net = nx.read_gpickle('data/cpnet.graph')
    vocab_set = set()
    with open('raw_files/glove.6B.300d.txt', 'r') as f:
        for line in f:
            word = line.rstrip('\n').split(' ')[0]
            vocab_set.add(word)
    concept2id = vocab_set.intersection(set(concept_net.nodes))

    print('loading conceptnet...')
    ns = multiprocessing.Manager().Namespace()
    ns.all_data = alldata
    ns.concept2id = concept2id
    ns.batch_size = int(len(ns.all_data) / 24) + 10
    # ns.batch_size = 5000
    mylist = [i for i in range(0, len(ns.all_data), ns.batch_size)]  # valid
    ns.concept_net = concept_net

    p = multiprocessing.Pool()
    func = partial(handle_func, ns)
    result = p.map(func, mylist)
    flat_graphs = []
    flat_res = []
    for p_result in result:
        for dialog, concepts, global_graph, easy_target, hard_target, entity_path in p_result:
            line = { 'dialog': dialog, 'concepts': concepts, 
                'easy_target': easy_target,
                'hard_target': hard_target,
                'entity_path': entity_path,
             }
            flat_res.append(line)
            flat_graphs.append(global_graph)

    avg_nodes = sum([ len(graph.nodes) for graph in flat_graphs ]) / len(flat_graphs)
    print(filedir, ' avg_nodes ', avg_nodes)
    pickle.dump(flat_graphs, open(filedir + 'graphs.pkl', 'wb'))

    with open(filedir + 'concepts_nv.json', 'w') as f:
        for line in flat_res:            
            json.dump(line, f)
            f.write('\n')

if __name__ == "__main__":
    main('data/TGConv/test/')
    main('data/TGConv/dev/') 
    main('data/TGConv/train/') 
    print('reasoning finished')

# nohup python -u preprocess/4tgconv_reason.py > common.log 2>&1 &
# data/TGConv/test/  avg_nodes  947.9627163781624
# data/TGConv/dev/  avg_nodes  885.7418839808408
# data/TGConv/train/  avg_nodes  891.3370263476398