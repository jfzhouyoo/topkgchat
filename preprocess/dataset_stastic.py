from html import entities
import json
from tqdm import tqdm
from functools import partial
from transformers import AutoTokenizer, AutoModel
import torch
import random

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to('cuda')
model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_dialog_similarity(all_data):
    all_cos = []
    with torch.no_grad():
        for dialog in tqdm(all_data):
            # Tokenize input texts
            encoded_input = tokenizer(dialog, padding=True, truncation=True, return_tensors="pt").to('cuda')
            model_output = model(**encoded_input, output_hidden_states=True, return_dict=True)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = torch.functional.F.normalize(sentence_embeddings, p=2, dim=1)
            dialog_cos = []
            window = 1
            for i in range(window, len(sentence_embeddings)):
                # cos = torch.cosine_similarity(sentence_embeddings[i].unsqueeze(0), sentence_embeddings[i-window:i, :]).mean().item()
                cos = torch.cosine_similarity(sentence_embeddings[i], sentence_embeddings[i-window], dim=0).item()
                dialog_cos.append(cos)
            all_cos.extend(dialog_cos)

    return torch.tensor(all_cos).mean()


def main(filedir):
    print('-' * 50)
    print('count ', filedir)
    dialogs = []
    entities = []
    with open(filedir + '/concepts_nv.json') as f:
        for row in f:
            data = json.loads(row)
            dialogs.append(data['dialog'])
            entities.append(data['concepts'])

    print('data_len: ', len(dialogs))
    total_uttr = [len(d) for d in dialogs ]
    total_word = [ len(u.split(' ')) for d in dialogs for u in d ]
    total_entity = [ len(row) for ent in entities for row in ent ]
    print('avg uttr : ', sum(total_uttr) / len(total_uttr))
    print('avg word : ', sum(total_word) / len(total_word))
    print('avg entities : ', sum(total_entity) / len(total_entity))

    res = get_dialog_similarity(dialogs)
    print("dialog avg similarity", res)


if __name__ == "__main__":
    main('data/otter/test')
    main('data/otter/dev')
    main('data/otter/train')
    main('data/convai/test')
    main('data/convai/dev')
    main('data/convai/train')

# python preprocess/dataset_stastic.py
