# -*- coding: utf-8 -*-
import random
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, progress
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from argparse import ArgumentParser
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sacrebleu.metrics import BLEU
from nlgeval import NLGEval
import json
import pickle
import sys

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

class GenModel(pl.LightningModule):
    def __init__(self, baseline_gpt=None, batch_size=None, dataset=None, lr=None, num_workers=None, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.baseline_gpt = baseline_gpt
        self.num_workers = num_workers
        self.lr = lr
        self.dataset = dataset
        self.save_hyperparameters()
        self.dec_tok = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-small")
        # config = json.load(open('config/gpt/config.json'))
        # config.n_layer =  6
        # self.decoder = GPT2LMHeadModel(config=GPT2Config(**config))
        self.decoder = GPT2LMHeadModel.from_pretrained('cache/gpt')
        self.dec_tok.add_special_tokens({ 'additional_special_tokens': ["<|target|>"] })
        self.decoder.resize_token_embeddings(len(self.dec_tok))
        self.key_preditor = None
        self.bleu = BLEU()
        self.nlg_eval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=[])
        # "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"
            
    def prepare_data(self):
        self.test_result = []
        self.test_refs = []
        self.predict_result = []
        self.predict_refs = []
        self.hits_list = []

    def setup(self, stage: str = None):
        if stage == 'fit':
            with open('data/TGConv/train/concepts_nv.json') as f:
                train_data_1 = [json.loads(row) for row in f]
            with open('data/OTTers/train/concepts_nv.json') as f:
                train_data_2 = [json.loads(row) for row in f]
            with open('data/TGConv/dev/concepts_nv.json') as f:
                valid_data_1 = [json.loads(row) for row in f]
            with open('data/OTTers/dev/concepts_nv.json') as f:
                valid_data_2 = [json.loads(row) for row in f]

            if self.dataset == 'tgconv':
                self.train_dataset = train_data_1
                self.val_dataset = valid_data_1
            elif self.dataset == 'ott':
                self.train_dataset = train_data_2
                self.val_dataset = valid_data_2
            else:
                train_data_1.extend(train_data_2)
                valid_data_1.extend(valid_data_2)
                self.train_dataset = train_data_1
                self.val_dataset = valid_data_1

            print(f"train_len: {len(self.train_dataset)}, valid_len: {len(self.val_dataset)}")
        else:
            with open('data/TGConv/test/concepts_nv.json') as f:
                test_data_1 = [json.loads(row) for row in f]
            with open('data/OTTers/test/concepts_nv.json') as f:
                test_data_2 = [json.loads(row) for row in f]

            if self.dataset == 'tgconv':
                if stage == 'predict':
                    self.test_dataset = [ (data, None) for data in test_data_1 ][:150]
                else:
                    self.test_dataset = test_data_1
            elif self.dataset == 'ott':
                if stage == 'predict':
                    test_ref = {} 
                    for data in test_data_2:
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
                    self.test_dataset = [ (val, None) for val in test_ref.values() ]
                else:
                    self.test_dataset = test_data_2
                # test_graphs = pickle.load(open('data/OTTers/test/graphs.pkl', 'rb'))
                # self.test_dataset = [(d, g) for d, g in zip(test_data_2, test_graphs) ]
            else:
                test_data_1.extend(test_data_2)
                self.test_dataset = test_data_1
            
            print(f"test_len: {len(self.test_dataset)}")

    def generate_from_predictor(self, preds, preds_topk, context):
        input_ids = []
        input_mask = []
        result = []
        for ktotal, (ktop1, ktop3), c in zip(preds, preds_topk, context):
            eos = self.dec_tok.eos_token
            sep = self.dec_tok.additional_special_tokens[0]
            prompt = sep.join(ktop3) + sep
            # prompt = ktop1 + sep
            # if len(ktotal) == 0 or len(ktotal) > 5:
            #     prompt = sep.join(ktop3) + sep
            # else:
            #     prompt = sep.join(ktotal) + sep
            prompt_id = self.dec_tok.encode(prompt, add_special_tokens=False)
            context_id = self.dec_tok.encode(eos + c + eos, add_special_tokens=False)
            input_ids.append(prompt_id + context_id)
            input_mask.append([1] * len(input_ids[-1]))

        pad_to_max_seq_len(input_ids, pad_token_id=self.dec_tok.eos_token_id, max_len=256)
        pad_to_max_seq_len(input_mask, pad_token_id=0, max_len=256)
    
        response_out = self.decoder.generate(
            input_ids=torch.tensor(input_ids).to(self.device),
            attention_mask=torch.tensor(input_mask).to(self.device),
            max_new_tokens=40,
            pad_token_id=self.dec_tok.eos_token_id,
            eos_token_id=self.dec_tok.eos_token_id,
            num_beams=3,
            # do_sample=True,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.0,
            length_penalty=1.0,
            early_stopping=True
        )
        res_out = response_out[:, len(input_ids[0]) - 1:]
        for row in res_out:
            result.append(self.dec_tok.decode(row, skip_special_tokens=True))

        return result

    def collate_fn(self, batch):
        dec_labels = []
        dec_inputs = []
        dec_mask = []
        refs = []
        for i, data in enumerate(batch):  # context kws target ans
            eos = self.dec_tok.eos_token
            sep = self.dec_tok.additional_special_tokens[0]

            dialog = data['dialog']
            if self.dataset == 'ott':
                context = [dialog[-1], dialog[0]]
                response = dialog[1]
                keywords = data['concepts'][1]
            else:
                sample_index = random.randint(1, len(dialog) - 1)
                context = dialog[:sample_index]
                if len(context) > 3:
                    context = context[-3:]
                response = dialog[sample_index]
                keywords = data['concepts'][sample_index]
                
            refs.append(response)
            random.shuffle(keywords)
            prompt = sep.join(keywords) + sep
            if self.baseline_gpt == 1:
                prompt = sep
            prompt_id = self.dec_tok.encode(prompt, add_special_tokens=False)

            context = eos + eos.join(context)
            context_id = self.dec_tok.encode(context, add_special_tokens=False)

            response = eos + response + eos
            response_id = self.dec_tok.encode(response, add_special_tokens=False)
            
            dec_inputs.append(prompt_id + context_id + response_id)
            dec_labels.append([-100] * len(prompt_id) + [-100] * len(context_id) + response_id)
            dec_mask.append([1] * len(dec_inputs[-1]))

        pad_to_max_seq_len(dec_inputs, pad_token_id=self.dec_tok.eos_token_id, max_len=256)
        pad_to_max_seq_len(dec_labels, pad_token_id=-100, max_len=256)
        pad_to_max_seq_len(dec_mask, pad_token_id=0, max_len=256)

        return {
            'dec_inputs': torch.tensor(dec_inputs),
            'dec_mask': torch.tensor(dec_mask),
            'dec_labels': torch.tensor(dec_labels),
            'refs': refs,
        }

    def collate_fn_predict(self, batch):
        predictor_batch = self.key_preditor.collate_fn(batch)
        return predictor_batch

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=self.collate_fn_predict)

    def forward(self, dec_inputs=None, dec_mask=None, dec_labels=None, refs=None):
        outputs = self.decoder(
            dec_inputs,
            attention_mask=dec_mask,
            labels=dec_labels,
        )
        return {'loss': outputs['loss'], 'logits': outputs['logits'] }

    def training_step(self, batch: tuple, batch_idx: int) -> dict:
        outputs = self(**batch)
        return outputs['loss']

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        outputs = self(**batch)
        return {'val_loss': outputs['loss'].item() }

    def validation_epoch_end(self, val_step_outputs: list) -> dict:
        print('\n\n', '-' * 100, '\n\n')
        print('validation_epoch_end')
        val_loss = [x['val_loss'] for x in val_step_outputs]
        print('val_loss %.4f \n' % torch.tensor(val_loss).mean().item())
        self.log('val_loss', torch.tensor(val_loss).mean().item())
        print('\n\n', '-' * 100, '\n\n')

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        input_ids = batch['dec_inputs']
        input_mask = batch['dec_mask']
        response_out = self.decoder.generate(
            input_ids=input_ids,
            attention_mask=input_mask,
            max_new_tokens=40,
            pad_token_id=self.dec_tok.eos_token_id,
            eos_token_id=self.dec_tok.eos_token_id,
            num_beams=3,
            # do_sample=True,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.0,
            length_penalty=1.0,
            early_stopping=True
        )
        res_out = response_out[:, input_ids.shape[-1]-1:]
        for row in res_out:
            self.test_result.append(self.dec_tok.decode(row, skip_special_tokens=True))
        self.test_refs.extend(batch['refs'])

    def test_epoch_end(self, predict_outputs=None):
        print('\n\n', '-' * 100, '\n\n')
        print('test_epoch_end')

        bleu = self.bleu.corpus_score(self.test_result, [ self.test_refs ])
        print('bleu ', bleu)
        result = self.nlg_eval.compute_metrics(ref_list=[self.test_refs], hyp_list=self.test_result)
        print('nlg_eval ', result)

        with open('output/' + self.dataset + '_test_output.txt', 'w') as f:
            for pred, ground_truth in zip(self.test_result, self.test_refs):            
                f.write('gdth: ' + ground_truth + '\n')
                f.write('pred: ' + pred + '\n')
                f.write('\n')
        print('\n\n', '-' * 100, '\n\n')
        return result

    def predict_step(self, batch: dict, batch_idx: int) -> dict:
        if self.dataset == 'ott':
            preds, preds_topk, preds01 = self.key_preditor.predict(
                batch['input_ids'], batch['input_mask'], 
                batch['global_graphs'], batch['target_graphs'],
            )
            eos = self.dec_tok.eos_token
            context = [ r['dialog'][-1] + eos + r['dialog'][0] for r in batch['raw_batch'] ]
            refs = [ r['ref'] for r in batch['raw_batch'] ]
            response = self.generate_from_predictor(preds, preds_topk, context)
            self.predict_result.extend(response)
            self.predict_refs.extend(refs)
            assert len(self.predict_result) == len(self.predict_refs)
        elif self.dataset == 'tgconv':
            preds, preds_topk = self.key_preditor.predict(
                batch['input_ids'], batch['input_mask'], 
                batch['global_graphs'], batch['target_graphs'],
                batch['candidates_nodes']
            )
            eos = self.dec_tok.eos_token
            context = [ eos.join(r['context']) for r in batch['raw_batch'] ]
            refs = [ r['ref'] for r in batch['raw_batch'] ]
            top1 = [ (None, [pred[0]]) for pred in preds_topk ]
            response = self.generate_from_predictor(preds, top1, context)
            self.predict_result.extend(response)
            self.predict_refs.extend(refs)
            assert len(self.predict_result) == len(self.predict_refs)
        for topk, raw_list in zip(preds_topk, batch['raw_keywords']):
            top1 = 1 if topk[0] in raw_list else 0
            top3 = len(set(topk[1]).intersection(set(raw_list))) / 3
            top1_all = 1/len(raw_list) if topk[0] in raw_list else 0
            top3_all = len(set(topk[1]).intersection(set(raw_list))) / len(raw_list)
            self.hits_list.append((top1 , top3, top1_all, top3_all))

        return 1

    def on_predict_epoch_end(self, predict_outputs=None):
        print('\n\n', '-' * 100, '\n\n')
        print('predict_epoch_end')

        print('hits@1', torch.tensor([ row[0] for row in self.hits_list ]).float().mean())
        print('hits@3', torch.tensor([ row[1] for row in self.hits_list ]).float().mean())
        print('hits@1 / all ', torch.tensor([ row[2] for row in self.hits_list ]).float().mean())
        print('hits@3 / all ', torch.tensor([ row[3] for row in self.hits_list ]).float().mean())

        pred_response = self.predict_result
        refs = self.predict_refs

        with open('output/' + self.dataset + '_predict_output.txt', 'w') as f:
            for pred, ref in zip(pred_response, refs):
                if isinstance(ref, list):
                    ref = '\n'.join(ref)
                f.write('gdth: ' + ref + '\n')
                f.write('pred: ' + pred + '\n')
                f.write('\n')

        if self.dataset == 'ott':
            ref1, ref2, ref3 = [], [], []
            for ref in refs:
                ref1.append(ref[0])
                ref2.append(ref[1] if len(ref) >= 2 else ref[0])
                ref3.append(ref[2] if len(ref) >= 3 else ref[-1])
            bleu = self.bleu.corpus_score(pred_response, [ref1.copy(), ref2.copy(), ref3.copy()])
            print('bleu ', bleu)
            result = self.nlg_eval.compute_metrics(ref_list=[ref1, ref2, ref3], hyp_list=pred_response)
            print('nlg_eval ', result)
        else:
            bleu = self.bleu.corpus_score(pred_response, [refs])
            print('bleu ', bleu)
            result = self.nlg_eval.compute_metrics(ref_list=[refs], hyp_list=pred_response)
            print('nlg_eval ', result)

        print('\n\n', '-' * 100, '\n\n')
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.001)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=3, verbose=True),
                "monitor": "val_loss",
            },
        }
        
if __name__ == '__main__':
    seed_everything(100, workers=True)

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--acc_batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dataset", type=str, default='ott')
    parser.add_argument("--run_predict", type=str, default=None)
    parser.add_argument("--key_model", type=str, default=None)
    parser.add_argument("--baseline_gpt", type=int, default=0)
    args = parser.parse_args()

    tb_logger = pl_loggers.TensorBoardLogger('logs_gen', name='')
    checkpoint_callback = ModelCheckpoint(
        filename='best',
        save_weights_only=True,
        save_last=True,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    
    bar_callback = progress.TQDMProgressBar(refresh_rate=50 if args.run_predict is None else 1)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=6, verbose=False, mode="min")

    model = GenModel(**vars(args))

    # model = model.load_from_checkpoint('logs_gen/version_0/checkpoints/best.ckpt', strict=True)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=100,
        logger=tb_logger,
        callbacks=[checkpoint_callback, bar_callback, early_stop_callback],
        gradient_clip_val=0.5,
        log_every_n_steps=25,
        accumulate_grad_batches=args.acc_batch,
        # val_check_interval=0.5,
        # detect_anomaly=True,
        # limit_val_batches=0.5,
        # amp_backend='apex',
        # precision=16,
    )
    # args.dataset = 'tgconv'
    # args.run_predict = 'tgconv_best'
    # args.key_model = 'version_6'
    if args.run_predict is not None:
        model = model.load_from_checkpoint('logs_gen/' + args.run_predict + '/checkpoints/best.ckpt', strict=True)
        if args.dataset == 'ott':
            from predictor_OTTers import KeywordPredictor
            model.key_preditor = KeywordPredictor.load_from_checkpoint('logs_otters/' + args.key_model + '/checkpoints/best.ckpt')
        elif args.dataset == 'tgconv':
            from predictor_TGConv import KeywordPredictor
            model.key_preditor = KeywordPredictor.load_from_checkpoint('logs_tgconv/' + args.key_model + '/checkpoints/best.ckpt')
        model.batch_size = 4
        trainer.predict(model)
    else:
        trainer.fit(model)
        trainer.test(model)


# nohup python -u code/keyword_generator.py --baseline_gpt 1 --dataset ott > keyword_generator.log 2>&1 &
# nohup python -u code/keyword_generator.py --dataset ott --key_model best > keyword_generator.log 2>&1 &
# nohup python -u code/keyword_generator.py --dataset tgconv > keyword_generator_tgconv.log 2>&1 &
# nohup python -u code/keyword_generator.py --dataset tgconv --run_predict tgconv_best --key_model version_6 > tgconv.log 2>&1 &