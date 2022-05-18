import json
import torch
import random
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pickle
from torch_geometric.utils.convert import from_networkx
from nltk import word_tokenize
from discriminator import Discriminator
from keyword_generator import GenModel
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import spacy
import networkx as nx

class Env(object):
    def __init__(self, token2id=None, id2token=None, global_planing=None):
        super().__init__()
        self.device = 'cuda'

        with open('data/TGConv/test/concepts_nv.json') as f:
            train_data = [json.loads(row) for row in f]
        # train_graphs = pickle.load(open('data/TGConv/train/graphs.pkl', 'rb'))
        self.train_dataset = [(d, None) for d, g in zip(train_data, train_data) ]
        with open('data/hard_target.json', encoding='utf-8') as f:
            self.hard_target_set = json.load(f)
        self.token2id = token2id
        self.id2token = id2token
        self.global_planing = global_planing

        self.disc: Discriminator = Discriminator.load_from_checkpoint('logs_discri/version_14/checkpoints/best.ckpt')
        self.disc.to('cuda')
        self.disc.eval()
        self.disc.freeze()
        self.generator: GenModel = GenModel.load_from_checkpoint('logs_gen/tgconv_best/checkpoints/best.ckpt')
        self.generator.to('cuda')
        self.generator.eval()
        self.generator.freeze()
        self.train_idx = -1
        self.pst: PorterStemmer = PorterStemmer() 
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])

    def reset(self, target_type=None):
        self.train_idx = self.train_idx + 1
        if self.train_idx >= len(self.train_dataset) - 1:
            self.train_idx = 0
        dialog, _  = self.train_dataset[self.train_idx]  # context ['','',''] kws

        if target_type == 'hard':
            self.target = random.choice(self.hard_target_set)
        else:
            self.target = dialog['easy_target']

        self.context: list = dialog['dialog'][:2]
        self.kws: list = dialog['concepts'][:2]
        
        graph = nx.Graph()
        for row in self.kws:
            for source in row:
                self.global_planing.find_path(source, self.target, graph)

        graph_nodes = list(graph.nodes)
        for node in graph_nodes:
            graph.nodes[node]['x'] = self.token2id[node]

        nodes_mapping = dict(zip(graph_nodes, range(0, len(graph_nodes))))

        all_paths_nodes = set()
        action_paths = []
        path_max_len = 0
        for row in self.kws:
            for source in row:
                if source in graph:
                    try:
                        poteintial = list(nx.all_shortest_paths(graph, source, self.target))
                        for path in poteintial:
                            if len(path) >= path_max_len:
                                action_paths = path
                                path_max_len = len(path)
                            [ all_paths_nodes.add(n) for n in path ]
                    except nx.NetworkXNoPath:
                        continue
        if len(all_paths_nodes)  == 0:
            return self.reset()
        self.action_space = action_paths

        self.t_graph = [ self.token2id[neibor] for neibor in list(graph.neighbors(self.target)) ]

        self.global_graph = from_networkx(graph)

        input_ids = []
        for text in self.context: 
            input_ids.extend([ self.token2id[token] for token in word_tokenize(text) ])

        return { 'input_ids': input_ids, 't_graph': self.t_graph, 'g_graph': self.global_graph, 'action_space': self.action_space }

    def step(self, action_words_idx: list):
        pred_topk = self.global_graph.x[action_words_idx].tolist()
        # topk_words = [ self.id2token[id] for id in pred_topk ]
        topk_words = [ self.id2token[pred_topk] ]
        self.kws.append(topk_words)

        eos = '<|endoftext|>'
        context = eos.join(self.context[-3:])
        response = self.generator.generate_from_predictor([None], [(None, topk_words)], [context])[0].lower()
        self.context.append(response)
        
        input_ids = []
        unk_count = 0
        for text in self.context[-3:]: 
            for token in word_tokenize(text):
                if token in self.token2id:
                    input_ids.append(self.token2id[token])
                else:
                    unk_count += 1
                    
        state = { 'input_ids': input_ids, 'action_space': self.action_space, 't_graph': self.t_graph, 'g_graph': self.global_graph }

        discri_context = ' [SEP] '.join(self.context[-3:])
        input_encoding = self.disc.tokenizer.encode(discri_context, return_token_type_ids=False, return_tensors="pt").to(self.device)
        output = self.disc.model(input_encoding)['logits']
        positive_prob = output.sigmoid()[:, 1].item()

        reward = round((positive_prob - 0.5) * 4.0, 4)

        response_token = [ t.lemma_ for t in self.nlp(response) ]

        if len(set(topk_words).intersection(set(response_token))) == 0:
            reward -= 1.0
        # if unk_count >= 0:
        #     reward -= 0.5 * unk_count

        stem_target = self.pst.stem(self.target)
        # finish
        if self.target in response_token \
            or stem_target in response_token \
                or self.target in response \
                    or stem_target in response:

            # if random.randint(1, 100) == 10:
            #     print('-' * 40)
            #     print(self.target)
            #     print(self.kws)
            #     print(self.context)
            #     print('-' * 40)

            return state, 4.0, 1, 0

        if len(self.context) >= 8:
            return state, 0, 1, 0

        return state, reward, 0, 0
