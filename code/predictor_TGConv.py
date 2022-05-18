# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import numpy as np
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from transformers import BertTokenizer, BertModel, BertConfig, AutoModel
from pytorch_lightning.callbacks import ModelCheckpoint, progress
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import sys
import json
from torch_geometric.utils.convert import from_networkx
import pickle
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import networkx as nx
from global_planing import GlobalPlaning
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from torch_geometric.nn import GCNConv
import random
from sacrebleu.metrics import BLEU
from nlgeval import NLGEval

class GCN_Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(300, 512)
        self.conv2 = GCNConv(512, 300)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return x
        
def pad_to_max_seq_len(arr, max_seq_len=None, pad_token_id=0, max_len=None):
    """
    a = [ [1, 2, 3], [1, 3] ]
    pad_to_max_seq_len(a, 5)
    a -> [[1, 2, 3, 0, 0], [1, 3, 0, 0, 0]]
    """
    if max_seq_len is None:
        max_seq_len = 0
        for sub_a in arr:
            if len(sub_a) >= max_seq_len:
                max_seq_len = len(sub_a)
    if max_len is not None:
        if max_seq_len > max_len:
            max_seq_len = max_len
    for index, text in enumerate(arr):
        seq_len = len(text)
        if seq_len < max_seq_len:
            padded_tokens = [
                pad_token_id for _ in range(max_seq_len - seq_len)
            ]
            new_text = text + padded_tokens
            arr[index] = new_text
        elif seq_len > max_seq_len:
            new_text = text[:max_seq_len]
            arr[index] = new_text
    return max_seq_len

class KeywordPredictor(pl.LightningModule):
    def __init__(self, batch_size=None, lr=None, num_workers=None, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.save_hyperparameters()

        token2id, id2token, weights = pickle.load(open('data/graph_embedding.pkl', 'rb'))
        self.token2id = token2id
        self.id2token = id2token
        self.pretrain_weights = weights
        self.node2embedding = torch.nn.Embedding(len(weights), 300).from_pretrained(torch.FloatTensor(np.array(weights)), freeze=False)
        self.gcn_enc = GCN_Net()

        self.tok: BertTokenizer = BertTokenizer.from_pretrained('cache/bert')
        self.config = BertConfig(
            vocab_size=self.tok.vocab_size,
            num_hidden_layers=4,
            num_attention_heads=4,
            hidden_size=256,
            intermediate_size=1024,
            max_position_embeddings=512,
        )
        self.context_enc = nn.GRU(300, 256, 1, bidirectional=True)
        # self.encoder = BertModel(self.config)

        self.fc = nn.Sequential(
            nn.Linear(300 + 300 + 256*2, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2),
        )
        self.common_metric = MetricCollection([
            Accuracy(),
            ConfusionMatrix(num_classes=2),
        ])
        self.pr_positive = MetricCollection([
            Precision(ignore_index=0),
            Recall(ignore_index=0),
            F1Score(ignore_index=0),
        ])
        self.pr_negative = MetricCollection([
            Precision(ignore_index=1),
            Recall(ignore_index=1),
            F1Score(ignore_index=1),
        ])
        self.bleu = BLEU()
        self.nlg_eval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=[]) # "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"
        self.test_result = []
        self.hits_list = []
        self.predict_result = []
        self.graph_counter = []
        self.global_planing: GlobalPlaning = GlobalPlaning()
        self.pst: PorterStemmer = PorterStemmer() 
        self.generator = None

    def setup(self, stage: str = None):
        if stage == 'fit':
            with open('data/TGConv/train/concepts_nv.json') as f:
                train_data = [json.loads(row) for row in f]
            train_graphs = pickle.load(open('data/TGConv/train/graphs.pkl', 'rb'))
            self.train_dataset = [(d, g) for d, g in zip(train_data, train_graphs) ]

            with open('data/TGConv/dev/concepts_nv.json') as f:
                valid_data = [json.loads(row) for row in f]
            val_graphs = pickle.load(open('data/TGConv/dev/graphs.pkl', 'rb'))
            self.val_dataset = [(d, g) for d, g in zip(valid_data, val_graphs) ]
            # self.train_dataset.extend([(d, g) for d, g in zip(valid_data, val_graphs) ])
            print(f"train_len: {len(train_data)}, valid_len: {len(valid_data)}")
        elif stage == "test":
            with open('data/TGConv/test/concepts_nv.json') as f:
                test_data = [json.loads(row) for row in f]
            test_graphs = pickle.load(open('data/TGConv/test/graphs.pkl', 'rb'))
            self.test_dataset = [(d, g) for d, g in zip(test_data, test_graphs) ]
        elif stage == "predict":
            with open('data/TGConv/test/concepts_nv.json') as f:
                test_data = [json.loads(row) for row in f]
            self.predict_dataset = [(d, None) for d in test_data ][:150]
            print(f"len predicting: {len(self.predict_dataset)}")
            
    def collate_fn(self, batch):
        seqs = []
        global_graphs = []
        raw_keywords = []
        graph_counter = []
        target_graphs = []
        candidates_nodes = []
        candidates_labels = []
        input_ids = []
        raw_batch = []

        for data, graph in batch:  # context kws target ans
            dialog = data['dialog']
            if len(dialog) <= 4:
                continue
            sample_index = random.randint(2, len(dialog) - 2)
            context = dialog[:sample_index]
            if len(context) > 3:
                context = context[-3:]
            data['context'] = context
            data['ref'] = dialog[sample_index + 1]

            s_c = [ c for c in data['concepts'][sample_index - 1] if c in self.token2id ]
            b_c = [ c for c in data['concepts'][sample_index] if c in self.token2id ]
            t_c = [ c for c in data['concepts'][sample_index + 1] if c in self.token2id ]
            if len(b_c) == 0 or len(t_c) == 0:
                continue
            if graph is None:
                data['s_c'] = s_c
                data['b_c'] = b_c
                data['t_c'] = t_c
                graph = self._get_potential_path(data)
                graph_counter.append((len(graph), len([ 1 for c in b_c if c not in graph ]), len(b_c)))

            graph_nodes = list(graph.nodes)
            if len(graph_nodes) < 3:
                raise ValueError('graph_nodes not enough')

            nodes_mapping = dict(zip(graph_nodes, range(0, len(graph_nodes))))
            label_idx = [ nodes_mapping[c] for c in b_c if c in nodes_mapping]
            if len(label_idx) == 0:
                print("no label idx ", b_c)
                continue

            two_hop_set = set()
            for c in s_c:
                two_hop_set.add(c)
                if c not in graph_nodes:
                    continue
                for one_hop_c in nx.neighbors(graph, c):
                    if one_hop_c not in graph_nodes:
                        continue
                    two_hop_set.add(one_hop_c)
                    for two_hop_c in nx.neighbors(graph, one_hop_c):
                        two_hop_set.add(two_hop_c)

            two_hop_set = list(two_hop_set)
            two_hop_set_idx = [ nodes_mapping[c] for c in two_hop_set if c in nodes_mapping]
            label_01 = torch.full((1, len(graph_nodes)), -100).squeeze(0).long()
            label_01[two_hop_set_idx] = 0
            label_01[label_idx] = 1
            candidates_labels.append(label_01)
            candidates_nodes.append(two_hop_set_idx)

            raw_keywords.append(b_c)
            token_id = []
            for text in context: 
                token_id.extend([ self.token2id[token] for token in word_tokenize(text) ])
            input_ids.append(token_id)

            seqs.append(' [SEP] '.join(context))

            for node in graph_nodes:
                if node not in self.token2id:
                    raise ValueError('node not in', node)
                graph.nodes[node]['x'] = self.token2id[node]
            global_graphs.append(from_networkx(graph))
            t_c_nodes = [ self.token2id[c] for c in t_c ]
            for c in t_c:
                if c in graph:
                    t_c_nodes.extend([ self.token2id[nei_c] for nei_c in list(graph.neighbors(c)) ])
            target_graphs.append(torch.tensor(t_c_nodes))
            raw_batch.append(data)

        # input_encoding = self.tok.batch_encode_plus(
        #     seqs,
        #     padding=True,
        #     max_length=128,
        #     truncation=True,
        #     return_tensors='pt',
        #     add_special_tokens=True,
        # )
        input_encoding = {'attention_mask': []}
        pad_to_max_seq_len(input_ids, max_len=128)
        return {
            'input_ids': torch.tensor(input_ids),
            # 'input_ids': input_encoding['input_ids'],
            'input_mask': input_encoding['attention_mask'],
            'global_graphs': global_graphs,
            'target_graphs': target_graphs,
            'candidates_nodes': candidates_nodes,
            'candidates_labels': candidates_labels,
            'graph_counter': graph_counter,
            'raw_keywords': raw_keywords,
            'raw_batch': raw_batch,
        }

    def _get_potential_path(self, dialog):
        graph = nx.Graph()
        temp_nodes = dialog['s_c'].copy()
        temp_nodes.extend(dialog['t_c'])
        # for c in temp_nodes:
        #     nei1 = list(set([self.pst.stem(n1) for n1 in list(self.global_planing.concept_net.neighbors(c)) ]))
        #     nei1 = [ n1 for n1 in nei1 if n1 in self.token2id and n1 in self.global_planing.concept_net ][:5]
        #     for n1 in nei1:
        #         graph.add_node(n1, x=n1)
        #         graph.add_edge(c, n1)

        #         nei2 = list(set([self.pst.stem(n2) for n2 in list(self.global_planing.concept_net.neighbors(n1)) ]))
        #         nei2 = [ n2 for n2 in nei2 if n2 in self.token2id and n1 in self.global_planing.concept_net ][:5]
        #         for n2 in nei2:
        #             graph.add_node(n2, x=n2)
        #             graph.add_edge(n1, n2)
        for s in dialog['s_c']:
            for t in dialog['t_c']:
                self.global_planing.find_path(s, t, graph)
        return graph

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def encode_state(self, input_ids, action_space=None, t_graph=None, g_graph=None):
        g_graph = g_graph.to(self.device)
        input_ids = torch.tensor([input_ids]).to(self.device)
        t_graph = torch.tensor(t_graph).to(self.device)

        context_embeding = self.node2embedding(input_ids)
        context_embed = self.context_enc(context_embeding.permute(1, 0, 2))[1]
        context_embed = context_embed.permute(1, 0, 2).reshape(-1)

        graph_x = self.node2embedding(g_graph.x)
        global_graph_embed = self.gcn_enc.encode(graph_x, g_graph.edge_index) # global graph

        repeat_context_embed = context_embed.repeat(len(g_graph.x), 1) # context

        target_graph_embed = self.node2embedding(t_graph).mean(0) # target graph 
        target_graph_embed = target_graph_embed.repeat(len(g_graph.x), 1)

        state = torch.cat((repeat_context_embed, global_graph_embed, target_graph_embed), 1)

        return state

    def predict(self, input_ids, input_mask, global_graphs, target_graphs, candidates_nodes):
        # context_outputs = self.encoder(
        #     input_ids=input_ids,
        #     attention_mask=input_mask,
        #     return_dict=True,
        # )[1]
        context_embeding = self.node2embedding(input_ids)
        context_outputs = self.context_enc(context_embeding.permute(1, 0, 2))[1]
        context_outputs = context_outputs.permute(1, 0, 2).reshape(input_ids.shape[0], -1)

        preds = []
        preds_topk = []
        preds01 = []
        for context_embed, t_graph, g_graph, two_hop_idx in zip(context_outputs, target_graphs, global_graphs, candidates_nodes):
            graph_x = self.node2embedding(g_graph.x)
            global_graph_embed = self.gcn_enc.encode(graph_x, g_graph.edge_index) # global graph

            repeat_context_embed = context_embed.repeat(len(g_graph.x), 1) # context

            target_graph_embed = self.node2embedding(t_graph).mean(0) # target graph 
            target_graph_embed = target_graph_embed.repeat(len(g_graph.x), 1)

            target_graph_embed = target_graph_embed + global_graph_embed # augment
            
            pred_logits = self.fc(torch.cat((repeat_context_embed, global_graph_embed, target_graph_embed), 1))

            # pred_logits = pred_logits[two_hop_idx]
            # graph_idx = g_graph.x[two_hop_idx]
            _, pred_topk = pred_logits.softmax(1)[:, 1].topk(3)
            pred_topk = g_graph.x[pred_topk].tolist()
            pred_ids = g_graph.x[pred_logits.argmax(1) == 1].tolist()
            preds.append([ self.id2token[id] for id in pred_ids ])
            preds_topk.append((
                self.id2token[pred_topk[0]],
                [ self.id2token[id] for id in pred_topk ],
            ))
        return preds, preds_topk

    def forward(self, 
        input_ids=None, input_mask=None, candidates_nodes=None,
        global_graphs=None, candidates_labels=None, target_graphs=None,
        raw_keywords=None, graph_counter=None, raw_batch=None,
    ):
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        # context_outputs = self.encoder(
        #     input_ids=input_ids,
        #     attention_mask=input_mask,
        #     return_dict=True,
        # )[1]  # ([batch_size, hidden_size ])
        context_embeding = self.node2embedding(input_ids)
        context_outputs = self.context_enc(context_embeding.permute(1, 0, 2))[1]
        context_outputs = context_outputs.permute(1, 0, 2).reshape(input_ids.shape[0], -1)

        loss = None
        preds = []
        reals = []
        for context_embed, t_graph, g_graph, label in zip(context_outputs, target_graphs, global_graphs, candidates_labels):
            graph_x = self.node2embedding(g_graph.x)
            global_graph_embed = self.gcn_enc.encode(graph_x, g_graph.edge_index) # global graph

            repeat_context_embed = context_embed.repeat(len(g_graph.x), 1) # context

            target_graph_embed = self.node2embedding(t_graph).mean(0) # target graph 
            target_graph_embed = target_graph_embed.repeat(len(g_graph.x), 1)
            # target_graph_embed = target_graph_embed + global_graph_embed # augment

            pred_logits = self.fc(torch.cat((global_graph_embed, repeat_context_embed, target_graph_embed), 1))

            if loss is None:
                loss = loss_fct(pred_logits, label)
            else:
                loss += loss_fct(pred_logits, label)

            preds.extend(pred_logits.softmax(1).argmax(1).tolist())
            reals.extend(label.tolist())

        return {'loss': loss, 'preds': torch.tensor(preds).to(self.device), 'reals': torch.tensor(reals).to(self.device) }

    def training_step(self, batch: tuple, batch_idx: int) -> dict:
        outputs = self(**batch)
        return outputs['loss']

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        outputs = self(**batch)
        preds = outputs['preds'][outputs['reals'] != -100]
        reals = outputs['reals'][outputs['reals'] != -100]
        self.common_metric.update(preds, reals)
        self.pr_positive.update(preds, reals)
        self.pr_negative.update(preds, reals)
        return {'val_loss': outputs['loss'].item()}

    def _print_compute_reset(self):
        acc = self.common_metric.compute()
        self.log('accuracy', acc['Accuracy'])
        print('common_metric', acc)
        self.common_metric.reset()

        pr = self.pr_positive.compute()
        print('pr_positive ', pr)
        self.log('precision', pr['Precision'])
        self.log('f1', pr['F1Score'])
        print('pr_negative ', self.pr_negative.compute())

        self.pr_positive.reset()
        self.pr_negative.reset()

    def validation_epoch_end(self, val_step_outputs: list) -> dict:
        print('\n\n', '-' * 100, '\n\n')
        print('validation_epoch_end')

        self._print_compute_reset()

        val_loss = [x['val_loss'] for x in val_step_outputs]
        print('val_loss %.4f \n' % torch.tensor(val_loss).mean().item())
        self.log('val_loss', torch.tensor(val_loss).mean().item())

        print('\n\n', '-' * 100, '\n\n')

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        outputs = self(**batch)
        preds = outputs['preds'][outputs['reals'] != -100]
        reals = outputs['reals'][outputs['reals'] != -100]
        self.common_metric.update(preds, reals)
        self.pr_positive.update(preds, reals)
        self.pr_negative.update(preds, reals)
        return self.common_metric

    def test_epoch_end(self, val_step_outputs: list) -> dict:
        print('\n\n', '-' * 100, '\n\n')
        print('test_epoch_end')
        self._print_compute_reset()
        print('\n\n', '-' * 100, '\n\n')
        
    def predict_step(self, batch: dict, batch_idx: int) -> dict:
        self.graph_counter.extend(batch['graph_counter'])
        preds, preds_topk = self.predict(
            batch['input_ids'], batch['input_mask'], batch['global_graphs'], batch['target_graphs'], batch['candidates_nodes']
        )

        for topk, raw_list in zip(preds_topk, batch['raw_keywords']):
            top1 = 1 if topk[0] in raw_list else 0
            top3 = len(set(topk[1]).intersection(set(raw_list))) / 3
            top1_all = 1/len(raw_list) if topk[0] in raw_list else 0
            top3_all = len(set(topk[1]).intersection(set(raw_list))) / len(raw_list)
            self.hits_list.append((top1 , top3, top1_all, top3_all))

        if self.generator:
            context = [ r['dialog'][0] for r in batch['raw_batch'] ]
            refs = [ r['ref'] for r in batch['raw_batch'] ]
            response = self.generator.generate_from_predictor(preds_topk, context)
            self.predict_result.append((response, refs))
        return self.common_metric

    def on_predict_epoch_end(self, predict_outputs: list):
        print('\n\n', '-' * 100, '\n\n')
        print('on_predict_epoch_end')

        print('graph_counter avg len ', torch.tensor([ row[0] for row in self.graph_counter ]).float().mean())
        print('graph_counter avg miss node ', torch.tensor([ row[1] for row in self.graph_counter ]).float().mean())
        print('graph_counter avg b_c ', torch.tensor([ row[2] for row in self.graph_counter ]).float().mean())

        print('hits@1 / all ', torch.tensor([ row[0] for row in self.hits_list ]).float().mean())
        print('hits@3 / all', torch.tensor([ row[1] for row in self.hits_list ]).float().mean())
        print('hits@1', torch.tensor([ row[2] for row in self.hits_list ]).float().mean())
        print('hits@3', torch.tensor([ row[3] for row in self.hits_list ]).float().mean())

        if self.generator:
            pred_response = [ r for batch_res in self.predict_result for r in batch_res[0]]
            refs = [ r for batch_res in self.predict_result for r in batch_res[1]]

            ref1, ref2, ref3 = [], [], []
            for ref in refs:
                ref1.append(ref[0])
                ref2.append(ref[0] if len(ref) < 2 else ref[1])
                ref3.append(ref[0] if len(ref) < 3 else ref[2])

            bleu = self.bleu.corpus_score(pred_response, refs)
            print('bleu ', bleu)
            result = self.nlg_eval.compute_metrics(ref_list=[ref1, ref2, ref3], hyp_list=pred_response)
            print('nlg_eval ', result)

            with open('output/generator_output.txt', 'w') as f:
                for pred, ref in zip(pred_response, refs):
                    f.write('\n'.join(ref) + '\n')
                    f.write('pred: ' + pred + '\n')
                    f.write('\n')
            self.predict_result = []
        print('\n\n', '-' * 100, '\n\n')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.001)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', verbose=True),
                "monitor": "f1",
            },
        }

if __name__ == '__main__':
    seed_everything(100, workers=True)

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--run_predict", type=str, default=None)
    args = parser.parse_args()

    tb_logger = pl_loggers.TensorBoardLogger('logs_tgconv', name='')
    checkpoint_callback = ModelCheckpoint(
        filename='best',
        save_weights_only=True,
        save_last=True,
        verbose=True,
        monitor='f1',
        mode='max'
    )

    bar_callback = progress.TQDMProgressBar(refresh_rate=50 if args.run_predict is None else 1)
    early_stop_callback = EarlyStopping(monitor="f1", min_delta=0.00, patience=6, verbose=False, mode="max")

    model = KeywordPredictor(**vars(args))

    trainer = pl.Trainer(
        gpus=[0],
        max_epochs=10,
        logger=tb_logger,
        callbacks=[checkpoint_callback, bar_callback, early_stop_callback],
        gradient_clip_val=0.5,
        log_every_n_steps=50,
    )
    # args.run_predict = 'best'
    # if args.run_predict is not None:
    #     model = model.load_from_checkpoint('logs_tgconv/' + args.run_predict + '/checkpoints/best.ckpt', strict=True)
    #     model.batch_size = 8
    #     trainer.predict(model)
    # else:
    trainer.fit(model)
    trainer.test(model)

# nohup python -u code/predictor_TGConv.py > predictor_TGConv.log 2>&1 &
# nohup python -u code/predictor_TGConv.py --run_predict best > run_predict.log 2>&1 &
# kill $(ps -ef | grep TGConv | tr -s ' ' | cut -d ' ' -f 2)