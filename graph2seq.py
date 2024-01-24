import sys
# モジュールのディレクトリのパスを追加
sys.path.append('/Users/yt/Prog/python/torch_venv/lib/python3.11/site-packages/')
#sys.path.append('/home/tabei/prog/python/python-pytorch/lib/python3.9/site-packages/')

import torch
import torch.nn as nn

from torch import Tensor

from torch_geometric.nn import GCNConv
from torch_geometric.data import data
import torch_geometric.data as data
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_max_pool
from torch_geometric.data import DataLoader

import torch.nn.functional as F
import torch.optim as optim

from torch_scatter import scatter_mean

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import to_dense_batch

from torch_geometric.nn import GATConv, global_mean_pool

import torch.nn.init as init

from torch_geometric.nn import SAGEConv

import random
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ComparableItem:
    def __init__(self, value, tensor_size):
        self.value = value  # float value
        self.tensor_size = tensor_size  # torch.Size

    def __lt__(self, other):
        return self.value < other.value

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):  # オブジェクトの文字列表現を返す、デバッグに便利
        return f"ComparableItem(value={self.value}, tensor_size={self.tensor_size})"

class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def is_empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        if not self.is_empty():
            return heapq.heappop(self.elements)[1]
        else:
            return "Priority Queue is empty"

    def peek(self):
        if not self.is_empty():
            return self.elements[0][1]
        else:
            return "Priority Queue is empty"

    def qsize(self):
        return len(self.elements)

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        self.embedding_size = embedding_size
        den = torch.exp(-torch.arange(0, embedding_size, 2) * math.log(10000) / embedding_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        embedding_pos = torch.zeros((maxlen, embedding_size))
        embedding_pos[:, 0::2] = torch.sin(pos * den)
        embedding_pos[:, 1::2] = torch.cos(pos * den)
        embedding_pos = embedding_pos.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('embedding_pos', embedding_pos)

    def forward(self, token_embedding: Tensor):
        return token_embedding + self.embedding_pos[: token_embedding.size(0), :] * math.sqrt(self.embedding_size)
#        return self.dropout(token_embedding + self.embedding_pos[: token_embedding.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_size = embedding_size
        
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long())

class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, heads=1):
        super(GATEncoder, self).__init__()
        self.num_layers = num_layers
        self.heads = heads
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        for _ in range(1, num_layers):  
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
            
        return x

def generate_square_subsequent_mask(seq_len, device, PAD_IDX1 = 0, PAD_IDX2 = 0):
    mask = (torch.triu(torch.ones((seq_len, seq_len), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill((mask == PAD_IDX1) | (mask == PAD_IDX2), float(0.0))
    return mask

class TransformerSequenceDecoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, output_dim, seq_len, nhead=8):
        super(TransformerSequenceDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len 
        self.num_layers = num_layers
        self.nhead = nhead
        self.embedding = TokenEmbedding(output_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=0.1)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, graph_embedding, target_sequence):
        memory = graph_embedding.unsqueeze(0).repeat(self.seq_len-1, 1, 1) 
        
        target_emb = self.embedding(target_sequence)
        target_emb = self.pos_encoder(target_emb) #seq_len, batch_size, hidden_dim

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.seq_len-1).to(target_emb.device)

        output = self.transformer_decoder(target_emb, memory, tgt_mask=tgt_mask)
        output = self.fc_out(output)

        return output

    def generate_sequence(self, graph_embedding, start_idx = 0, end_idx = 1):
        memory = graph_embedding.unsqueeze(0).repeat(self.seq_len-1, 1, 1) 

        ys = torch.ones(1, 1).fill_(start_idx).type(torch.long).to(graph_embedding.device)
        for i in range(self.seq_len - 1):
            target_emb = self.embedding(ys)
            target_emb = self.pos_encoder(target_emb)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(0)).to(ys.device)

            output = self.transformer_decoder(target_emb, memory, tgt_mask=tgt_mask)
            output = self.fc_out(output)
            output = output.transpose(0, 1)    
            output = output[:, -1]


            k = 3  # 例として上位3つの値を取得
            values, indices = torch.topk(output, k)

            _, next_word = torch.max(output, dim = 1)
            next_word = next_word.item()

            next_word_tensor = torch.ones(1, 1).type(torch.long).fill_(next_word).to(ys.device)
            ys = torch.cat([ys, next_word_tensor], dim=0)
            
            if next_word == end_idx:
                break

        return ys

    def generate_topk(self, graph_embedding, topk=5, start_idx = 0, end_idx = 1, beam_width=5, prune_ratio=0.7):
        memory = graph_embedding.unsqueeze(0).repeat(self.seq_len-1, 1, 1) 
        pq = PriorityQueue()
        ys = torch.ones(1, 1).fill_(start_idx).type(torch.long).to(graph_embedding.device)
        item = ComparableItem(0.0, ys)
        pq.put(item, -item.value)
        res = []
        max_score = float('-inf')
        
        while (pq.is_empty() == False):
            item = pq.get()

#            print("item:", item)

            score, seq = item.value, item.tensor_size
            normalized_score = score / len(seq)

            if score < 0.0:
                max_score = max(max_score, normalized_score)
            
            if len(seq) == self.seq_len:
                res.append([score, seq])
                continue
            
#            print("max_score:", max_score)
#            print("score:", score)
#            print("max_score * (1 + prune_ration):", max_score * (1 + prune_ratio))
#            print("normalized_score:", normalized_score)

            if normalized_score <= max_score * (1 + prune_ratio):
#                print("pruning")
                continue
#            print("item:", item)
#            print("item.value:", item.value)        
#            print("item.tensor_size:", item.tensor_size)
#            print(len(item.tensor_size))
            
            target_emb = self.embedding(seq)
            target_emb = self.pos_encoder(target_emb)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq.size(0)).to(seq.device)
            output = self.transformer_decoder(target_emb, memory, tgt_mask=tgt_mask)
            output = self.fc_out(output)
            output = output.transpose(0, 1)    
            output = output[:, -1]
            output = F.softmax(output, dim=-1)
            output = torch.log(output)

            scores, indices = torch.topk(output, topk)

    #            print("scores:", scores)
            
            for i, (new_score, index) in enumerate(zip(scores[0], indices[0])):
                next_word_tensor = torch.ones(1, 1).type(torch.long).fill_(index).to(seq.device)
                new_seq = torch.cat([seq, next_word_tensor], dim=0)
                if index != end_idx:
                    item = ComparableItem(score + new_score.item(), new_seq)
                    pq.put(item, -item.value)
                else:
                    res.append([score + new_score.item(), new_seq])
            
        return res

class Graph2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers_gate, output_dim, num_layers, seq_len, heads):
        super(Graph2Seq, self).__init__()
        self.encoder = GATEncoder(input_dim, hidden_dim, num_layers_gate, heads)
        self.decoder = TransformerSequenceDecoder(hidden_dim * heads, num_layers, output_dim, seq_len)

    def forward(self, x, edge_index, batch, tgt):
#        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_embeddings = self.encoder(x, edge_index)
        graph_embedding = global_mean_pool(node_embeddings, batch)

        return self.decoder(graph_embedding, tgt)

    def generate_sequence(self, x, edge_index, batch):  
        node_embeddings = self.encoder(x, edge_index)
        graph_embedding = global_mean_pool(node_embeddings, batch)  # (batch_size, hidden_dim)
        return self.decoder.generate_sequence(graph_embedding)

    def generate_topk(self, x, edge_index, batch, topk=5):
        node_embeddings = self.encoder(x, edge_index)
        graph_embedding = global_mean_pool(node_embeddings, batch)
        return self.decoder.generate_topk(graph_embedding, topk)