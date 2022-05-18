# -*- coding: utf-8 -*-
import os
from stage2_ppo import A2CPolicyNetwork
import torch
from tqdm import tqdm
from pytorch_lightning import seed_everything
import json
import random
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='new')
    parser.add_argument("--target", default="easy", type=str)
    args = parser.parse_args()

    eos = '<|endoftext|>'
    seed_everything(0, workers=True)
    device = 'cuda:0'
    print("Loading model...")
    model: A2CPolicyNetwork = A2CPolicyNetwork.load_from_checkpoint('logs_rl/version_0/checkpoints/last.ckpt')
    model.to(device)
        

    counter = []
    test_len = 100
    with torch.no_grad():
        all_conversation = []
        for i in tqdm(range(0, test_len)):
            done = 0
            state_dic = model.env.reset(args.target)
            target = model.env.target

            real_state = model.predictor.encode_state(**state_dic).to(device)
            action_space = state_dic['action_space']
            logits = model.predictor.fc(real_state)
            action_idx = logits.argmax(1).topk(3)[1][0].item()
            for node in action_space:
                context = eos.join(model.env.context[-3:])
                response = model.env.generator.generate_from_predictor([None], [(None, [node])], [context])[0].lower()
                model.env.context.append(response)
                model.env.kws.append(node)
                if target in response:
                    break
                if len(model.env.context) >= 8:
                    break
            all_conversation.append((model.env.context, action_space, target, target in response))
            counter.append(len(model.env.context) - 2)

    print('avg len ', sum(counter) / test_len)

    with open('output/' + args.target + '_simulation_output.txt', 'w') as fout:
        for conv in all_conversation:
            fout.write('\nTarget: ' + conv[2])
            fout.write(' Finish \n'  if conv[-1] is True  else ' Failed \n')
            fout.write('Path: ' + '-'.join(conv[1]) + '\n')
            fout.write('\n'.join(conv[0]) + '\n\n')

# python code/rl_test.py --target easy
# python code/rl_test.py --target hard