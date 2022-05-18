import pickle
import networkx as nx
import os
import numpy as np
from tqdm import tqdm

def load_total_vec(concept2id=None):
    if os.path.exists('conceptnet/numberbatch.pkl'):
        with open('conceptnet/numberbatch.pkl', 'rb') as f:
            print('load pkl')
            return pickle.load(f)
    graph_embed = {}
    with open('conceptnet/numberbatch-en-19.08.txt') as f:
        total_len = int(f.readline().split(' ')[0])
        for line in tqdm(f, total = total_len):
            line = line.rstrip('\n').split(' ')
            if len(line) != 301:
                raise ValueError('vector error')
            word = line[0]
            # if word.replace('#', '') in self.concept2id:
            #     print('specail word', word)
            if word not in concept2id:
                continue
            graph_embed[word] = np.array(line[1:], dtype=np.float64)

    with open('conceptnet/numberbatch.pkl', 'wb') as f:
        pickle.dump(graph_embed, f)
    return graph_embed

def get_cos_similar_multi(v1: list, v2: list):
    num = np.dot([v1], np.array(v2).T)  
    denom = np.linalg.norm(v1) * np.linalg.norm(v2, axis=1)
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

class GlobalPlaning():
    def __init__(self, MAX_STEP=3):
        self.concept_net = nx.read_gpickle('data/cpnet.graph')
        token2id, id2token, weights = pickle.load(open('data/graph_embedding.pkl', 'rb'))
        self.graph_token2id = token2id
        self.graph_id2token = id2token
        self.graph_embed = weights
        self.MAX_STEP = MAX_STEP
        self.start_embed = None
        self.target_embed = None

    def getEmbed(self, token):
        return self.graph_embed[self.graph_token2id[token]]

    def getWords(self, word, cur_path):
        words_list = []
        neighbors_word = []
        neighbors_embed = []
        if word not in self.graph_token2id:
            return []
        for w in list(self.concept_net.neighbors(word)):
            if w in self.graph_token2id and w not in cur_path:
                if sum(self.getEmbed(w)) == 0:
                    continue
                neighbors_word.append(w)
                neighbors_embed.append(self.getEmbed(w))
        if len(neighbors_embed) == 0:
            return []
        smooth_score = get_cos_similar_multi(self.start_embed, neighbors_embed).reshape(-1)
        words_list = [(w, round(cos, 4)) for w, cos in zip(neighbors_word, smooth_score)]
        sorted_word_list1 = sorted(words_list, key=lambda pair: pair[1], reverse=True)[:10]
        for pair in sorted_word_list1:
            self.sub_g.add_edge(word, pair[0])

        forward_score = get_cos_similar_multi(self.target_embed, neighbors_embed).reshape(-1)
        words_list = [(w, round(cos, 4)) for w, cos in zip(neighbors_word, forward_score)]
        sorted_word_list2 = sorted(words_list, key=lambda pair: pair[1], reverse=True)[:10]
        for pair in sorted_word_list2:
            self.sub_g.add_edge(word, pair[0])
            
        sorted_word_list1.extend(sorted_word_list2)
        return sorted_word_list1

    def tree_search(self, word, target_word, cur_path, edges):
        if word and (len(cur_path) >= self.MAX_STEP or word == target_word):
            path_vector = ""
            for v in cur_path:
                path_vector = path_vector + v + " "
            edges.append(path_vector.rstrip(' '))
            return edges

        new_choices = self.getWords(word, cur_path)

        if len(new_choices) == 0:
            path_vector = ""
            for v in cur_path:
                path_vector = path_vector + v + " "
            edges.append(path_vector.rstrip(' '))

        for new_word in new_choices:
            # new_word:[word,cos]
            if new_word[0] in cur_path:
                continue
            cur_path.append(new_word[0])
            self.tree_search(new_word[0], target_word, cur_path, edges)
            cur_path.pop()
        return edges

    def find_path(self, start_word, end_word, sub_g):
        if start_word not in self.graph_token2id or end_word not in self.graph_token2id:
            return []
        self.sub_g = sub_g
        res_paths = []
        self.start_embed = self.getEmbed(start_word)
        self.target_embed = self.getEmbed(end_word)
        cur_path = [start_word]
        edges = self.tree_search(start_word, end_word, cur_path, [])
        res_paths.extend(edges)

        self.start_embed = self.getEmbed(end_word)
        self.target_embed = self.getEmbed(start_word)
        cur_path = [end_word]
        edges = self.tree_search(end_word, start_word, cur_path, [])
        res_paths.extend(edges)
        # try:
        #     res = list(nx.all_simple_edge_paths(self.sub_g, source=start_word, target=end_word, cutoff=2))
        # except:
        #     print("can not find ", start_word, end_word)
        #     res = []
        return res_paths

if __name__ == '__main__':
    test = GlobalPlaning()
    # res = test.find_path('bick', 'bien')
    # print(res)
    # res = test.get_node_by_keyword([['favorite', 'type', 'south'], ['moonlight', 'kind', 'dancing', 'music']], 'adventure')
    # print(res)
    sub_g = nx.Graph()
    res = test.find_path('art', 'outside', sub_g)
    print(res)
    nx.all_shortest_paths(sub_g, 'art', 'outside')
    res = test.find_path('dance', 'computer')
    print(res)
