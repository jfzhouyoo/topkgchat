# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, progress, EarlyStopping
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torchmetrics import Accuracy, MetricCollection, Precision, Recall, ConfusionMatrix
import numpy as np
import json
import random
import argparse
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import pickle

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class Discriminator(pl.LightningModule):
    def __init__(self, learning_rate=None, num_workers=None, batch_size=None, **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.save_hyperparameters()
        self.tokenizer = BertTokenizer.from_pretrained('cache/bert')
        self.model = BertForSequenceClassification.from_pretrained('cache/bert', num_labels=2)
        self.metrics = MetricCollection([
            Accuracy(),
            ConfusionMatrix(num_classes=2),
            Precision(ignore_index=0),
            Recall(ignore_index=0),
        ])
        self.sent_tok = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.sent_enc = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to('cuda')
        self.sent_enc.eval()

    def prepare_data(self):
        with open('data/TGConv/train/concepts_nv.json') as f:
            self.train_data = [json.loads(row) for row in f]
        with open('data/TGConv/dev/concepts_nv.json') as f:
            self.valid_data = [json.loads(row) for row in f]
        with open('data/TGConv/test/concepts_nv.json') as f:
            self.test_data = [json.loads(row) for row in f]
        id = 0
        for dataset in [ self.train_data, self.valid_data, self.test_data ]:
            for data in dataset:
                data['id'] = id
                id += 1
        if os.path.exists('cache/uttr_embed.pkl'):
            print('cache/uttr_embed.pkl exist')
            uttr_position2embed, uttr_position2uttr = pickle.load(open('cache/uttr_embed.pkl', 'rb'))
            self.uttr_position2embed = uttr_position2embed
            self.uttr_position2uttr = uttr_position2uttr
            return

        corpus = []
        self.uttr_position2embed = {}
        self.uttr_position2uttr = {}
        for dataset in [ self.train_data, self.valid_data, self.test_data ]:
            for data in dataset:
                for idx, uttr in enumerate(data['dialog']):
                    key = str(data['id']) + '_' + str(idx)
                    corpus.append((uttr, key))
                    self.uttr_position2uttr[key] = uttr

        batch = [i for i in range(0, len(corpus), 300)]
        with torch.no_grad():
            for start_idx in tqdm(batch, desc="pre calc sematic semilarity"):
                batch_data = corpus[start_idx: start_idx+300]
                uttrs = [d for (d, key) in batch_data ]
                uttrs_keys = [key for (d, key) in batch_data ]
                encoded_input = self.sent_tok(uttrs, padding=True, truncation=True, return_tensors="pt").to('cuda')
                model_output = self.sent_enc(**encoded_input, output_hidden_states=True, return_dict=True)
                sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                sentence_embeddings = torch.functional.F.normalize(sentence_embeddings, p=2, dim=1)
                for key, s in zip(uttrs_keys, sentence_embeddings):
                    self.uttr_position2embed[key] = s.to('cpu')

        with open('cache/uttr_embed.pkl', 'wb') as f:
            print('cache/uttr_embed.pkl saved')
            pickle.dump((self.uttr_position2embed, self.uttr_position2uttr), f)

    def collate_fn(self, batch):
        eos = ' [SEP] '
        labels = []
        seqs = []

        all_keys = list(self.uttr_position2embed.keys())

        for data in batch:
            dialog = data['dialog']
            if len(dialog) <= 3:
                continue
            sample_index = random.randint(3, len(dialog))
            positive_sample = dialog[sample_index-3:sample_index]
            
            response_position = str(data['id']) + '_' + str(sample_index-1)
            response_embed = self.uttr_position2embed[response_position]
            
            candidates_keys = random.sample(all_keys, 500)
            candidates_embed = torch.stack([self.uttr_position2embed[c] for c in candidates_keys])
            candidates_text = [self.uttr_position2uttr[c] for c in candidates_keys]

            cosine_sim = torch.cosine_similarity(response_embed.unsqueeze(0), candidates_embed)
            topk_choice = random.choice(cosine_sim.topk(5)[1].tolist())
            negative_sample = positive_sample.copy()
            negative_sample[-1] = candidates_text[topk_choice]

            seqs.append(eos.join(positive_sample))
            seqs.append(eos.join(negative_sample))
            labels.append(1)
            labels.append(0)

        input_encoding = self.tokenizer.batch_encode_plus(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            max_length=128,
        )
        return {
            'input_ids': input_encoding['input_ids'],
            'attention_mask': input_encoding['attention_mask'],
            'labels': torch.tensor(labels),
        }

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.valid_data,  shuffle=False, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_data,  shuffle=False, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def forward(self, input_ids, input_mask=None, labels=None, weights=None):
        output = self.model(
            input_ids,
            attention_mask=input_mask,
        )['logits']
        loss = None
        positive_prob = output.sigmoid()[:, 1]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(output, labels)
        return {'prob': positive_prob, 'loss': loss, 'pred': output.argmax(1) }

    def training_step(self, batch: tuple, batch_idx: int):
        outputs = self(batch['input_ids'], input_mask=batch['attention_mask'], labels=batch['labels'])
        self.log('train_loss', outputs['loss'])
        return outputs['loss']

    def validation_step(self, batch: tuple, batch_idx: int):
        outputs = self(batch['input_ids'], input_mask=batch['attention_mask'], labels=batch['labels'])
        self.metrics.update(outputs['pred'], batch['labels'])
        return outputs['loss']

    def validation_epoch_end(self, val_step_outputs=None):
        val_loss = torch.tensor(val_step_outputs).mean()
        self.log('val_loss', val_loss.item())
        score = self.metrics.compute()
        print('metrics', score)
        self.log('acc', score['Accuracy'])
        self.metrics.reset()

    def test_step(self, batch: tuple, batch_idx: int):
        outputs = self(batch['input_ids'], input_mask=batch['attention_mask'], labels=batch['labels'])
        self.metrics.update(outputs['pred'], batch['labels'])

    def test_epoch_end(self, test_step_outputs=None):
        score = self.metrics.compute()
        print('testing metrics', score)
        self.log('acc', score['Accuracy'])
        self.metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.001)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', verbose=True),
                "monitor": "val_loss",
            },
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='discriminator')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=8e-5)
    args = parser.parse_args()

    seed_everything(20, workers=True)
    tb_logger = pl_loggers.TensorBoardLogger('logs_discri/', name='')
    checkpoint_callback = ModelCheckpoint(
        filename='best',
        save_weights_only=True,
        save_last=True,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    bar_callback = progress.TQDMProgressBar(refresh_rate=100)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min")

    model = Discriminator(**vars(args))

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=10,
        logger=tb_logger,
        detect_anomaly=True,
        gradient_clip_val=0.5,
        callbacks=[checkpoint_callback, bar_callback],
    )
    trainer.fit(model)
    trainer.test(model)

# nohup python -u code/discriminator.py > discriminator.log 2>&1 &
# kill $(ps -ef | grep discriminator | tr -s ' ' | cut -d ' ' -f 2)
