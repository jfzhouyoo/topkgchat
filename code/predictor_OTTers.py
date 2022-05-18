# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import numpy as np
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from transformers import BertTokenizer, BertModel, BertConfig
from pytorch_lightning.callbacks import ModelCheckpoint, progress
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import json
from torch_geometric.utils.convert import from_networkx
import pickle
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
    def __init__(self, batch_size=None, plan_type=None, lr=None, num_workers=None, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.plan_type = plan_type
        self.lr = lr
        self.save_hyperparameters()

        token2id, id2token, weights = pickle.load(open('data/graph_embedding.pkl', 'rb'))
        self.token2id = token2id
        self.id2token = id2token
        self.pretrain_weights = weights
        self.node2embedding = torch.nn.Embedding(len(weights), 300).from_pretrained(torch.FloatTensor(np.array(weights)), freeze=False)
        self.gcn_enc = GCN_Net()

        self.tok: BertTokenizer = BertTokenizer.from_pretrained('config/bert')
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
            with open('data/OTTers/train/concepts_nv.json') as f:
                train_data = [json.loads(row) for row in f]
            train_graphs = pickle.load(open('data/OTTers/train/graphs.pkl', 'rb'))
            self.train_dataset = [(d, g) for d, g in zip(train_data, train_graphs) ]

            with open('data/OTTers/dev/concepts_nv.json') as f:
                valid_data = [json.loads(row) for row in f]
            val_graphs = pickle.load(open('data/OTTers/dev/graphs.pkl', 'rb'))
            self.val_dataset = [(d, g) for d, g in zip(valid_data, val_graphs) ]
            # self.train_dataset.extend([(d, g) for d, g in zip(valid_data, val_graphs) ])
            random.shuffle(self.train_dataset)
            print(f"train_len: {len(train_data)}, valid_len: {len(valid_data)}")
        elif stage == "test":
            with open('data/OTTers/test/concepts_nv.json') as f:
                test_data = [json.loads(row) for row in f]
            test_graphs = pickle.load(open('data/OTTers/test/graphs.pkl', 'rb'))
            self.test_dataset = [(d, g) for d, g in zip(test_data, test_graphs) ]
        elif stage == "predict":
            # with open('data/OTTers/test/concepts_nv.json') as f:
            #     test_data = [json.loads(row) for row in f]
            # self.predict_dataset = [(d, None) for d in test_data ]
            test_ref = {} 
            with open('data/OTTers/test/concepts_nv.json') as f:
                for row in f:
                    data = json.loads(row)
                    ref = data['dialog'][1]
                    s_t = ','.join(data['s_c'] + data['t_c'])
                    if s_t in test_ref:
                        test_ref[s_t]['ref'].append(ref)
                        test_ref[s_t]['b_c'].extend(data['b_c'])
                    else:
                        test_ref[s_t] = {
                            'idx': len(test_ref.keys()),
                            's_c': data['s_c'],
                            't_c': data['t_c'],
                            'b_c': data['b_c'].copy(),
                            'dialog': data['dialog'],
                            'ref': [ref],
                        }
            self.predict_dataset = [ (val, None) for val in test_ref.values() ]
            print(f"len predicting: {len(self.predict_dataset)}")
            
    def collate_fn(self, batch):
        seqs = []
        global_graphs = []
        label_word_idx = []
        raw_keywords = []
        graph_counter = []
        target_graphs = []
        input_ids = []
        raw_batch = []

        for dialog, graph in batch:  # context kws target ans
            b_c = [ c for c in dialog['b_c'] if c in self.token2id ]
            t_c = [ c for c in dialog['t_c'] if c in self.token2id ]
            if len(b_c) == 0 or len(t_c) == 0:
                continue
            if graph is None:
                graph = self._get_potential_path(dialog)
                graph_counter.append((len(graph), len([ 1 for c in b_c if c not in graph ]), len(b_c)))

            nodes = list(graph.nodes)
            if len(nodes) < 3:
                raise ValueError('nodes not enough')

            raw_keywords.append(list(set(b_c)))
            raw_batch.append(dialog)

            context = dialog['dialog']
            seqs.append(context[0] + ' [SEP] ' + context[-1])
            token_id = [ self.token2id[token] for token in word_tokenize(context[0]) ]
            # token_id.extend([ self.token2id[token] for token in word_tokenize(context[0]) ])
            input_ids.append(token_id)

            nodes_mapping = dict(zip(nodes, range(0, len(nodes))))
            label_idx = [ nodes_mapping[c] for c in b_c if c in nodes_mapping]
            label_word_idx.append(torch.zeros(len(nodes)).long())
            label_word_idx[-1][label_idx] = 1
            for node in nodes:
                if node not in self.token2id:
                    raise ValueError('node not in', node)
                graph.nodes[node]['x'] = self.token2id[node]
            global_graphs.append(from_networkx(graph))
            t_c_nodes = [ self.token2id[c] for c in t_c ]
            for c in t_c_nodes:
                if c in graph:
                    t_c_nodes.extend([ self.token2id[nei_c] for nei_c in list(graph.neighbors(c)) ])
            target_graphs.append(torch.tensor(t_c_nodes))

        input_encoding = self.tok.batch_encode_plus(
            seqs,
            padding=True,
            max_length=64,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True,
        )
        pad_to_max_seq_len(input_ids, max_len=64)
        return {
            'input_ids': torch.tensor(input_ids),
            # 'input_ids': input_encoding['input_ids'],
            'input_mask': input_encoding['attention_mask'],
            'global_graphs': global_graphs,
            'target_graphs': target_graphs,
            'graph_word_labels': label_word_idx,
            'graph_counter': graph_counter,
            'raw_keywords': raw_keywords,
            'raw_batch': raw_batch,
        }

    def _get_potential_path(self, dialog):
        graph = nx.Graph()
        temp_nodes = dialog['s_c'].copy()
        temp_nodes.extend(dialog['t_c'])
        NEIBOR_NUM = 10
        if self.plan_type == 'large_graph':
            NEIBOR_NUM = 20
        for c in temp_nodes:
            nei1 = list(set([self.pst.stem(n1) for n1 in list(self.global_planing.concept_net.neighbors(c)) ]))
            nei1 = [ n1 for n1 in nei1 if n1 in self.token2id and n1 in self.global_planing.concept_net ][:NEIBOR_NUM] #  
            for n1 in nei1:
                graph.add_node(n1, x=n1)
                graph.add_edge(c, n1)

                nei2 = list(set([self.pst.stem(n2) for n2 in list(self.global_planing.concept_net.neighbors(n1)) ]))
                nei2 = [ n2 for n2 in nei2 if n2 in self.token2id and n2 in self.global_planing.concept_net ][:NEIBOR_NUM] #
                for n2 in nei2:
                    graph.add_node(n2, x=n2)
                    graph.add_edge(n1, n2)
        if self.plan_type != 'no_plan':
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

    def predict(self, input_ids, input_mask, global_graphs, target_graphs):
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
        for context_embed, t_graph, g_graph in zip(context_outputs, target_graphs, global_graphs):
            graph_x = self.node2embedding(g_graph.x)
            global_graph_embed = self.gcn_enc.encode(graph_x, g_graph.edge_index) # global graph

            repeat_context_embed = context_embed.repeat(len(g_graph.x), 1) # context

            target_graph_embed = self.node2embedding(t_graph).mean(0) # target graph 
            target_graph_embed = target_graph_embed.repeat(len(g_graph.x), 1)

            target_graph_embed = target_graph_embed + global_graph_embed # augment
            
            pred_logits = self.fc(torch.cat((repeat_context_embed, global_graph_embed, target_graph_embed), 1))

            preds01.append(pred_logits.argmax(1).tolist())
            _, pred_topk = pred_logits.softmax(1)[:, 1].topk(3)
            pred_topk = g_graph.x[pred_topk].tolist()
            pred_ids = g_graph.x[pred_logits.argmax(1) == 1].tolist()
            preds.append([ self.id2token[id] for id in pred_ids ])
            preds_topk.append((
                self.id2token[pred_topk[0]],
                [ self.id2token[id] for id in pred_topk ],
            ))
        return preds, preds_topk, preds01

    def forward(self, 
        input_ids=None, input_mask=None, 
        global_graphs=None, graph_word_labels=None, target_graphs=None,
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
        for context_embed, t_graph, g_graph, label in zip(context_outputs, target_graphs, global_graphs, graph_word_labels):
            graph_x = self.node2embedding(g_graph.x)
            global_graph_embed = self.gcn_enc.encode(graph_x, g_graph.edge_index) # global graph

            repeat_context_embed = context_embed.repeat(len(g_graph.x), 1) # context

            target_graph_embed = self.node2embedding(t_graph).mean(0) # target graph 
            target_graph_embed = target_graph_embed.repeat(len(g_graph.x), 1)

            target_graph_embed = target_graph_embed + global_graph_embed # augment
            
            pred_logits = self.fc(torch.cat((repeat_context_embed, global_graph_embed, target_graph_embed), 1))

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
        self.common_metric.update(outputs['preds'], outputs['reals'])
        self.pr_positive.update(outputs['preds'], outputs['reals'])
        self.pr_negative.update(outputs['preds'], outputs['reals'])
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
        self.common_metric.update(outputs['preds'], outputs['reals'])
        self.pr_positive.update(outputs['preds'], outputs['reals'])
        self.pr_negative.update(outputs['preds'], outputs['reals'])
        return self.common_metric

    def test_epoch_end(self, val_step_outputs: list) -> dict:
        print('\n\n', '-' * 100, '\n\n')
        print('test_epoch_end')
        self._print_compute_reset()
        print('\n\n', '-' * 100, '\n\n')
        
    def predict_step(self, batch: dict, batch_idx: int) -> dict:
        self.graph_counter.extend(batch['graph_counter'])
        preds, preds_topk, preds01 = self.predict(
            batch['input_ids'], batch['input_mask'], batch['global_graphs'], batch['target_graphs'],
        )

        for topk, raw_list in zip(preds_topk, batch['raw_keywords']):
            top1 = 1 if topk[0] in raw_list else 0
            top3 = len(set(topk[1]).intersection(set(raw_list))) / 3
            top1_all = 1/len(raw_list) if topk[0] in raw_list else 0
            top3_all = len(set(topk[1]).intersection(set(raw_list))) / len(raw_list)
            self.hits_list.append((top1 , top3, top1_all, top3_all))

        preds01_flatten = torch.tensor([ n for row in preds01 for n in row ]).to(self.device)
        real_flatten = torch.tensor([ n for row in batch['graph_word_labels'] for n in row.tolist() ]).to(self.device)

        self.common_metric.to(self.device).update(preds01_flatten, real_flatten)
        self.pr_positive.to(self.device).update(preds01_flatten, real_flatten)
        self.pr_negative.to(self.device).update(preds01_flatten, real_flatten)

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

        print('hits@1 ', torch.tensor([ row[0] for row in self.hits_list ]).float().mean())
        print('hits@3', torch.tensor([ row[1] for row in self.hits_list ]).float().mean())
        print('hits@1 / all', torch.tensor([ row[2] for row in self.hits_list ]).float().mean())
        print('hits@3 / all', torch.tensor([ row[3] for row in self.hits_list ]).float().mean())

        print('common_metric', self.common_metric.compute())

        print('pr_positive ', self.pr_positive.compute())
        print('pr_negative ', self.pr_negative.compute())

        if self.generator:
            pred_response = [ r for batch_res in self.predict_result for r in batch_res[0]]
            refs = [ r for batch_res in self.predict_result for r in batch_res[1]]

            ref1, ref2, ref3 = [], [], []
            for ref in refs:
                ref1.append(ref[0])
                ref2.append(ref[0] if len(ref) < 2 else ref[1])
                ref3.append(ref[0] if len(ref) < 3 else ref[2])

            bleu = self.bleu.corpus_score(pred_response, [refs])
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
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', verbose=True),
                "monitor": "val_loss",
            },
        }

if __name__ == '__main__':
    seed_everything(100, workers=True)

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--run_predict", type=str, default=None)
    parser.add_argument("--plan_type", type=str, default=None)
    # parser.add_argument("--run_predict", type=str, default='logs_otters/version_1/checkpoints/best.ckpt')
    args = parser.parse_args()

    tb_logger = pl_loggers.TensorBoardLogger('logs_otters', name='')
    checkpoint_callback = ModelCheckpoint(
        filename='best',
        save_weights_only=True,
        save_last=True,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    bar_callback = progress.TQDMProgressBar(refresh_rate=25 if args.run_predict is None else 1)

    model = KeywordPredictor(**vars(args))

    # model.load_from_checkpoint('logs_otters/version_15/checkpoints/best.ckpt')

    trainer = pl.Trainer(
        gpus=[0],
        max_epochs=10,
        logger=tb_logger,
        callbacks=[checkpoint_callback, bar_callback],
        gradient_clip_val=0.5,
        log_every_n_steps=25,
    )

    # args.run_predict = 'version_0'
    if args.run_predict is not None:
        model = model.load_from_checkpoint('logs_otters/' + args.run_predict + '/checkpoints/best.ckpt', strict=True)
        model.batch_size = 8
        trainer.predict(model)
    else:
        trainer.fit(model)
        trainer.test(model)
        model.batch_size = 8
        trainer.predict(model) # use the global planning graph

# nohup python -u code/predictor_OTTers.py > predictor_OTTers.log 2>&1 &
# nohup python -u code/predictor_OTTers.py --plan_type 'no_plan' > predictor_OTTers.log 2>&1 &
# nohup python -u code/predictor_OTTers.py --plan_type 'large_graph' > predictor_OTTers.log 2>&1 &
# nohup python -u code/predictor_OTTers.py --run_predict version_3 > predictor_OTTers.log 2>&1 &

# kill $(ps -ef | grep TGConv | tr -s ' ' | cut -d ' ' -f 2)

# graph_counter avg len  tensor(1073.6783)
# graph_counter avg miss node  tensor(2.6865)
# graph_counter avg b_c  tensor(12.1432)
# hits@1 / all  tensor(0.6892)
# hits@3 / all tensor(0.6387)
# hits@1 tensor(0.0687)
# hits@3 tensor(0.1832)
# common_metric {'Accuracy': tensor(0.9941), 'ConfusionMatrix': tensor([[394063,    554],
#         [  1796,    848]])}
# pr_positive  {'Precision': tensor(0.6049), 'Recall': tensor(0.3207), 'F1Score': tensor(0.4192)}
# pr_negative  {'Precision': tensor(0.9955), 'Recall': tensor(0.9986), 'F1Score': tensor(0.9970)}