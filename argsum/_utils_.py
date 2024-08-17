# Import packages
import requests
import json

import torch
import torch.nn as nn
import traceback
from transformers import BartTokenizer, BartForConditionalGeneration
from typing import List

import numpy as np
import pandas as pd

# Define functions
def set_summetix_api_login(username, password):
    # Create dict
    api_login = {'username':username, 'password':password}
    # Save login data
    with open('argsum/___summetix_login.json', 'w') as f:
        json.dump(api_login, f)

def get_summetix_api_login():
    # Get API login 
    with open('argsum/___summetix_login.json') as f:
        login = json.load(f)

    # Set url and payload
    api_url = 'https://api.summetix.com/en/get_api_key'
    payload = login

    # Get API data
    response = requests.post(api_url, data = json.dumps(payload), headers = {'Content-Type': 'application/json'}).json()

    # Save API data
    with open('argsum/___summetix_api_login.json', 'w') as f:
        json.dump(response, f)

def get_summetix_api_key_id():
    # Load API key and and id
    with open('argsum/___summetix_api_login.json') as f:
        api_key_id = json.load(f)
        api_key = api_key_id['apiKey']
        api_id = api_key_id['apiKeyId']
    return api_key, api_id

def load_test_df(name):
    if name == 'ArgKP21':
        df = pd.read_csv('data/ArgKP-2021/dataset_splits_scores_processed.csv')
        df_test = df[df['set'] == 'test'].sort_values(by = ['topic', 'key_point', 'stance', 'argument'])
    elif name == 'Debate_test':
        df_test = pd.read_csv('data/Debate/dataset_scores_processed_test.csv')
    # Return dataframe 
    return df_test

# Define objects
class BARTScorer:
    def __init__(self, device='mps', max_length=1024, checkpoint='facebook/bart-large-cnn'):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path=None):
        """ Load model from paraphrase finetuning """
        if path is None:
            path = 'models/metrics/bart_score.pth'
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score(self, cands, refs, batch_size=4):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(cands), batch_size):
            src_list = cands[i: i + batch_size]
            tgt_list = refs[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list

    def multi_ref_score(self, cands, refs: List[List[str]], agg="mean", batch_size=4):
        # Assert we have the same number of references
        ref_nums = [len(x) for x in refs]
        if len(set(ref_nums)) > 1:
            raise Exception("You have different number of references per test sample.")

        ref_num = len(refs[0])
        score_matrix = []
        for i in range(ref_num):
            curr_refs = [x[i] for x in refs]
            scores = self.score(cands, curr_refs, batch_size)
            score_matrix.append(scores)
        if agg == "mean":
            score_list = np.mean(score_matrix, axis=0)
        elif agg == "max":
            score_list = np.max(score_matrix, axis=0)
        else:
            raise NotImplementedError
        return list(score_list)

    def test(self, batch_size=3):
        """ Test """
        src_list = [
            'This is a very good idea. Although simple, but very insightful.',
            'Can I take a look?',
            'Do not trust him, he is a liar.'
        ]

        tgt_list = [
            "That's stupid.",
            "What's the problem?",
            'He is trustworthy.'
        ]

        print(self.score(src_list, tgt_list, batch_size))