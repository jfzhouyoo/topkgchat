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
sys.path.append('../')

def handle_func(ns, start_idx):
    print("handle for every conversation...")
    batch_data = ns.all_data[start_idx: start_idx + ns.batch_size]
    concept_net = ns.concept_net
    concept2id = ns.concept2id
    pst = PorterStemmer() 

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
        return ans

    graphs = []
    for dialog in tqdm(batch_data, mininterval=30):
        # sentence s_c b_c t_c
        sub_graph = nx.Graph()
        add_edges(dialog['s_c'], dialog['s_c'], sub_graph)
        add_edges(dialog['b_c'], dialog['b_c'], sub_graph)
        add_edges(dialog['t_c'], dialog['t_c'], sub_graph)
        add_edges(dialog['s_c'], dialog['b_c'], sub_graph)
        add_edges(dialog['b_c'], dialog['t_c'], sub_graph)

        temp_nodes = dialog['s_c'].copy()
        temp_nodes.extend(dialog['b_c'])
        temp_nodes.extend(dialog['t_c'])
        for c in temp_nodes:
            nei1 = list(set([pst.stem(n1) for n1 in list(concept_net.neighbors(c)) ]))
            random.shuffle(nei1)
            nei1 = [ n1 for n1 in nei1 if n1 in concept2id ][:10]
            for n1 in nei1:
                sub_graph.add_node(n1, x=n1)
                sub_graph.add_edge(c, n1)

                nei2 = list(set([pst.stem(n2) for n2 in list(concept_net.neighbors(n1)) ]))
                random.shuffle(nei2)
                nei2 = [ n2 for n2 in nei2 if n2 in concept2id ][:10]
                for n2 in nei2:
                    sub_graph.add_node(n2, x=n2)
                    sub_graph.add_edge(n1, n2)
        graphs.append(sub_graph)
    return graphs


def main(filedir):
    with open(filedir + 'concepts_nv.json') as f:
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
    graphs = [ graph for res in result for graph in res ]
    avg_nodes = sum([ len(graph.nodes) for res in result for graph in res ]) / len(graphs)
    print(filedir, ' avg_nodes ', avg_nodes)
    pickle.dump(graphs, open(filedir + 'graphs.pkl', 'wb'))

if __name__ == "__main__":
    main('data/OTTers/train/') 
    main('data/OTTers/dev/') 
    main('data/OTTers/test/')
    print('reasoning finished')

# nohup python -u preprocess/3otter_reason.py > common.log 2>&1 &
# 30 neibors
# data/OTTers/train/  avg_nodes  1906.7222222222222
# data/OTTers/dev/  avg_nodes  1963.4027777777778
# data/OTTers/test/  avg_nodes  1867.9982300884956
# 10 neibors
# data/OTTers/train/  avg_nodes  686.1499508357915
# data/OTTers/dev/  avg_nodes  705.0998263888889
# data/OTTers/test/  avg_nodes  671.1451327433629