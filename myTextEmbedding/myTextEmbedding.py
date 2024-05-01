import torch
import torch.nn as nn
from torch import tensor 
from transformers import BertModel, BertTokenizer
import gzip
import pandas as pd
import requests


class EmbeddingModel(nn.Module):
    def __init__(self, bertName = "bert-base-uncased"): # other bert models can also be supported
        super().__init__()
        self.bertName = bertName
        # use BERT model
        self.tokenizer = BertTokenizer.from_pretrained(self.bertName)
        self.model = BertModel.from_pretrained(self.bertName)        
       
    def forward(self, s, device = "cuda"):
        # get tokens, which also include attention_mask
        tokens = self.tokenizer(s, return_tensors='pt', padding = "max_length", truncation = True, max_length = 256).to(device)
        
        # get token embeddings
        output = self.model(**tokens)
        tokens_embeddings = output.last_hidden_state
        #print("tokens_embeddings:" + str(tokens_embeddings.shape))
        
        # mean pooling to get text embedding
        embeddings = tokens_embeddings * tokens.attention_mask[...,None] # [B, T, emb]
        #print("embeddings:" + str(embeddings.shape))
        
        embeddings = embeddings.sum(1) # [B, emb]
        valid_tokens = tokens.attention_mask.sum(1) # [B]
        embeddings = embeddings / valid_tokens[...,None] # [B, emb]    
        
        return embeddings

    # from scratch: nn.CosineSimilarity(dim = 1)(q,a)
    def cos_score(self, q, a): 
        q_norm = q / (q.pow(2).sum(dim=1, keepdim=True).pow(0.5))
        r_norm = a / (a.pow(2).sum(dim=1, keepdim=True).pow(0.5))
        return (q_norm @ r_norm.T).diagonal()
    
# contrastive training
class TrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = EmbeddingModel("bert-base-uncased")

    def forward(self, s1, s2, score):        
        cos_score = self.m.cos_score(self.m(s1), self.m(s2))
        loss = nn.MSELoss()(cos_score, score)
        return loss, cos_score
    
def searchWiki(s):
    response = requests.get(
            'https://en.wikipedia.org/w/api.php',
            params={
                'action': 'query',
                'format': 'json',
                'titles': s,
                'prop': 'extracts',
                'exintro': True,
                'explaintext': True,
            }
        ).json()
    page = next(iter(response['query']['pages'].values()))
    return page['extract'].replace("\n","")

# sentence chunking
def chunk(w):
    return w.split(".")

def generate_chunk_data(concepts):
    wiki_data = [searchWiki(c).replace("\n","") for c in concepts]
    chunk_data = []
    for w in wiki_data:
        chunk_data = chunk_data + chunk(w) 

    chunk_data = [c.strip()+"." for c in chunk_data]
    while '.' in chunk_data:
        chunk_data.remove('.')
    
    return chunk_data

def generate_chunk_emb(m, chunk_data):
    return m(chunk_data)

def search_document(s, chunk_data, chunk_emb, m, topk=3):
    question = [s]
    result_score = m.cos_score(m(question).expand(chunk_emb.shape),chunk_emb)
    print(result_score)
    _,idxs = torch.topk(result_score,topk)
    print([result_score.flatten()[idx] for idx in idxs.flatten().tolist()])
    return [chunk_data[idx] for idx in idxs.flatten().tolist()]

