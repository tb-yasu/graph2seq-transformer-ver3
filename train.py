import sys
#sys.path.append('/home/ytabei/prog/python-venv/lib/python3.10/site-packages/')
sys.path.append('/Users/yt/Prog/python/torch_venv/lib/python3.11/site-packages/')

import time
import argparse
import pickle

from data_loader import load_data_from_json
from utils import get_model_parameters
from utils import save_model
from graph2seq import Graph2Seq
from utils import train
from utils import generate_test_topk

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup

import torch.nn.functional as F

def train_model(model, train_data_iter, query_data_iter, epochs=2000, batch_size=4000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device : " + str(device))
    
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.001)
#    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
#    scheduler = ExponentialLR(optimizer, gamma=0.9)
#    scheduler = ReduceLROnPlateau(optimizer, 'min')

    # トレーニングパラメータ
    num_training_steps = epochs
    num_warmup_steps = epochs / 10

    # スケジューラの初期化
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=num_warmup_steps, 
                                                num_training_steps=num_training_steps)

    criterion = torch.nn.CrossEntropyLoss()

    dataloader = DataLoader(train_data_iter, batch_size=batch_size, shuffle=True)


    iter = 0
    best_model = model
    best_loss = 100000
    for epoch in range(epochs):
        train_loss = train(model, dataloader, scheduler, optimizer, criterion)
#        query_loss = evaluate(model, query_data_iter, criterion)
#        print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Test Loss: {query_loss}")
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}")

#        generate_test_topk(model, train_data_iter)
        
        if best_loss > train_loss:
            best_loss = train_loss
            best_model = model
            iter = 0
        elif iter == 5:
            iter = 0
        else:
            iter += 1


def main():
    parser = argparse.ArgumentParser(description="Train a graph2seq-pytorch model")
    parser.add_argument('--input_file', type=str, help="input file name")
    parser.add_argument('--output_file', type=str, help="output file name")
    parser.add_argument('--num_layers', type=int, help="num_layers", default=10)
    parser.add_argument('--num_layers_gate', type=int, help="num_layers_gate", default=3)
    parser.add_argument('--hidden_dim',  type=int, help="hidden_dim", default=128)
    parser.add_argument('--heads', type=int, help="heads", default=5)
    parser.add_argument('--epochs', type=int, help="epochs", default=2000)
    parser.add_argument('--batch_size', type=int, help="batch size", default=4000)
    args = parser.parse_args()

    print("Train graph2seq-pytorch model")
    print("input_file:", args.input_file)
    print("output_file:", args.output_file)
    print("num_layers:", args.num_layers)
    print("num_layers_gate:", args.num_layers_gate)
    print("hidden_dim:", args.hidden_dim)
    print("heads:", args.heads)

    cids, data_list, token_to_id, id_to_token, max_node_label, max_edge_label = load_data_from_json(args.input_file)
    input_dim, output_dim, seq_len = get_model_parameters(data_list)

    print("max_node_label:", max_node_label)
    print("max_edge_label:", max_edge_label)
    print("input_dim:", input_dim)
    print("output_dim:", output_dim)
    print("start_token_idx:", token_to_id['+'])
    print("end_token_idx:", token_to_id['*'])

    print("Training graph2seq model")
    start_time = time.time()
    model = Graph2Seq(input_dim, args.hidden_dim, args.num_layers_gate, output_dim, args.num_layers, seq_len, args.heads)
    train_model(model, data_list, args.epochs, args.batch_size)
    end_time = time.time()

    elapsed_time = end_time - start_time  # 実行時間を計算
    print("Execution time: {:.2f}sec".format(elapsed_time))

    print("Saving the learned model to the output_file: ", args.output_file)
    save_model(model, args.output_file)

    print("Saving the associated outputs to the file", args.output_file + "_dic.pkl")
    with open(args.output_file + "_dic.pkl", 'wb') as f:
        pickle.dump(token_to_id, f)
        pickle.dump(id_to_token, f)
        pickle.dump(max_node_label, f)        
        pickle.dump(max_edge_label, f)
        pickle.dump(input_dim, f)
        pickle.dump(args.hidden_dim, f)
        pickle.dump(output_dim, f)
        pickle.dump(args.num_layers, f)
        pickle.dump(args.num_layers_gate, f)
        pickle.dump(seq_len, f)
        pickle.dump(args.heads, f)


if __name__ == "__main__":
    main()