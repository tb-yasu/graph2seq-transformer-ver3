import sys
import time
import argparse
import pickle

from data_loader import load_data_from_json
from utils import get_model_parameters
from utils import save_model
from utils import load_model
from graph2seq import Graph2Seq

from utils import predict_sequence
from utils import predict_sequence_topk

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Train a graph2seq-pytorch model")
    parser.add_argument('--input_file', type=str, help="input file name")    
    parser.add_argument('--model_file', type=str, help="model file name")
    parser.add_argument('--output_file', type=str, help="output file name")
    parser.add_argument('--topk', type=int, help="topk", default=1)
    args = parser.parse_args()

    print("Predict")
    print("input_file:", args.input_file)
    print("model_file:", args.model_file)
    print("output_file:", args.output_file)
    print("topk:", args.topk)

    token_to_id = {}
    id_to_token = {}
    max_node_label = 0
    max_edge_label = 0
    input_dim = 0
    output_dim = 0
    num_layers = 0
    num_layers_gate = 0
    seq_len = 0
    heads = 0
    with open(args.model_file + "_dic.pkl", 'rb') as f:
        token_to_id  = pickle.load(f)
        id_to_token  = pickle.load(f)
        max_node_label   = pickle.load(f)
        max_edge_label  = pickle.load(f)
        input_dim    = pickle.load(f)
        hidden_dim   = pickle.load(f)
        output_dim   = pickle.load(f)
        num_layers   = pickle.load(f)
        num_layers_gate = pickle.load(f)
        seq_len      = pickle.load(f)
        heads        = pickle.load(f)

#    print("token_to_id:", token_to_id)
#    print("id_to_token:", id_to_token)
    print("max_node_label:", max_node_label)
    print("max_edge_label:", max_edge_label)
    print("input_dim:", input_dim)
    print("hidden_dim:", hidden_dim)
    print("output_dim:", output_dim)
    print("num_layers:", num_layers)
    print("num_layers_gate", num_layers_gate)
    print("seq_len:", seq_len)
    print("heads:", heads)
    print("start_token_idx:", token_to_id['+'])
    print("end_token_idx:", token_to_id['*'])

    print("load model:", args.model_file)
    model = Graph2Seq(input_dim, hidden_dim, num_layers_gate, output_dim, num_layers, seq_len, heads)
    load_model(model, args.model_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    qcids, query_list, token_to_id, id_to_token, max_node_label, max_edge_label = load_data_from_json(args.input_file, token_to_id, id_to_token, max_node_label, max_edge_label)

    dataloader = DataLoader(query_list, batch_size=1, shuffle=False)

    total_time = 0.0
    iter = 0
    with open(args.output_file, "w") as file:
        for batch_data in dataloader:
            start_time = time.process_time()
            qcid = qcids[iter]
            iter += 1
            print(qcid)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sample_graph = batch_data.to(device)

            x, edge_index, batch = sample_graph.x, sample_graph.edge_index, sample_graph.batch

            pred_seq = predict_sequence(model, x, edge_index, batch)
            
            print("qcid: ", qcid)
            pred_seq_cpu = pred_seq.to('cpu')
            token = ''
            for elem in pred_seq_cpu:
                token += id_to_token[int(elem.item())] + ' '
            print("predicted sequence")
            print(token)
#            token = ''
#            for i in pred_seq:
#                token += id_to_token[int(i)] + ' '
#            print(token)

#            token = ''
#            for i in pred_seq.item():
                #print(f"Element {i}: {pair}")       
#                token += (id_to_token[int(i)] + ' ')
#            print("predicted sequence 1 with score ")
#            print(token)

#            print("args.topk:", args.topk)
#            topk_pairs = predict_sequence_topk(model, sample_graph, max_length, args.topk)
#            print("topk_pairs:", topk_pairs)
#            print("len:", len(topk_pairs))
#            end_time = time.process_time()
#            total_time += (end_time - start_time)
#            iter += 1
#            idx = 0
#            print("qcid: ", qcid)
#            for predicted_seq, score in topk_pairs:
#                predicted_seq_cpu = predicted_seq.cpu()
#                print("predicted_seq_cpu shape:", predicted_seq_cpu.shape)
#                print("predicted_seq_cpu:", predicted_seq_cpu)
#                values = predicted_seq_cpu.tolist()  # [0]を使用して外側のリストを取り除く
#                tokens = ''
#                for id in values:
#                    tokens += (id_to_token[int(id)] + ' ')
#                print(f"Predicted sequence {idx + 1} with score {score.item():.4f}:")
#                print(tokens)
#                idx += 1


            print("Actual sequence:")
            tokens = ''
            for id in sample_graph.y.tolist():
                tokens += (id_to_token[int(id)] + ' ')
            print(tokens)
            print()

    ave_time = total_time / iter        
    print(f"Average CPU Time: {ave_time} seconds")

if __name__ == "__main__":
    main()