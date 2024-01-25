
import torch

def get_model_parameters(data_list):
    data = data_list[0]  # 最初のDataオブジェクトを取得
    input_dim = len(data.x[0])  # ノードの特徴ベクトルの次元数
    seq_len = len(data.y)  # 出力シーケンスの長さ

    unique_elements = set()
    for data in data_list:
        unique_elements.update(data.y.tolist())
    unique_count = len(unique_elements)

    output_dim = unique_count
    
    return input_dim, output_dim, seq_len

def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    
def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
    
def model_size(model):
    num_params = sum(p.numel() for p in model.parameters())
    # 通常のパラメータはfloat32で、それぞれ4バイトです。
    return num_params * 4 / (1024 ** 2)  # MB単位で返す

def train(model, dataloader, scheduler, optimizer, criterion):
    max_sequence_length = 7
    alpha = 0.9  # 適宜調整
    weights = torch.tensor([alpha ** t for t in range(max_sequence_length)], dtype=torch.float32)
    
#    model.train()
    seq_len = 7
    total_loss = 0.0
#    for batch_data in tqdm(dataloader):
    for batch_data in dataloader:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_data = batch_data.to(device)  # データをGPUに移動

        tgt = batch_data.y 
        tgt = tgt.view(-1, seq_len)
        tgt = tgt.transpose(0,1)
            
        input_tgt = tgt[:-1, :]

        if batch_data.edge_index.shape == torch.Size([0]):
            continue

        output = model(batch_data.x, batch_data.edge_index, batch_data.batch, input_tgt)

        optimizer.zero_grad()

        output_tgt = tgt[1:, :] # batch, seq_len

        sequence_length = output.shape[0]
        batch_size = output.shape[1]
        loss = 0.0

        for t in range(sequence_length):
            # 出力とラベルの選択
            output_t = output[t]  # output_t の形状: [batch_size, num_classes]
            label_t = output_tgt[t]    # label_t の形状: [batch_size]

            # 重みを取得
            weight = weights[t]

            # 重み付き損失を計算
            loss += weight * criterion(output_t, label_t)

        loss /= sequence_length

#        loss = criterion(output.reshape(-1, output.shape[-1]), output_tgt.reshape(-1))

        loss.backward()     
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        
        total_loss += loss.item()

    scheduler.step()

    return total_loss / len(dataloader)
    
def evaluate(model, dataloader, criterion):
    model.eval()
    seq_len = 7
    total_loss = 0.0
    for batch_data in dataloader:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_data = batch_data.to(device)  # データをGPUに移動

        tgt = batch_data.y 
        tgt = tgt.view(-1, seq_len)
        tgt = tgt.transpose(0,1)
            
        input_tgt = tgt[:-1, :]

        if batch_data.edge_index.shape == torch.Size([0]):
            continue

        output = model(batch_data.x, batch_data.edge_index, batch_data.batch, input_tgt)

        output_tgt = tgt[1:, :] # batch, seq_len

        loss = criterion(output.reshape(-1, output.shape[-1]), output_tgt.reshape(-1))
            
        total_loss += loss.item()

    return total_loss / len(dataloader)

def predict_sequence(model, x, edge_index, batch):
    model.eval()  # モデルを評価モードに設定
    with torch.no_grad():  # 勾配の計算を無効化
        output = model.generate_sequence(x, edge_index, batch)
        return output

def generate_test(model, dataloader):
    iter = 0
    model.eval()
    seq_len = 7
    total_loss = 0.0
    for batch_data in query_data:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sample_graph = batch_data.to(device)
        x, edge_index, batch = sample_graph.x, sample_graph.edge_index, sample_graph.batch

        pred_seq = predict_sequence(model, x, edge_index, batch)
    
#        print("qcid: ", qcid)
        pred_seq_cpu = pred_seq.to('cpu')
        token = ''
        for elem in pred_seq_cpu:
            token += id_to_token[int(elem.item())] + ' '
        print("predicted sequence")
        print(token)

    
        print("Actual sequence:")
        tokens = ''
        for id in sample_graph.y.tolist():
            tokens += (id_to_token[int(id)] + ' ')
        print(tokens)
        print()

        iter += 1
        if iter == 5:
            break

def predict_sequence_topk(model, x, edge_index, batch, topk=5):
    model.eval()  # モデルを評価モードに設定
    with torch.no_grad():  # 勾配の計算を無効化
        output = model.generate_topk(x, edge_index, batch, topk)
        return output

def generate_test_topk(model, dataloader):
    topk = 5
    iter = 0
    model.eval()
    seq_len = 7
    total_loss = 0.0
    for batch_data in query_data:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sample_graph = batch_data.to(device)
        x, edge_index, batch = sample_graph.x, sample_graph.edge_index, sample_graph.batch
#        pred_seq = predict_sequence(model, x, edge_index, batch)
        pred_seq_topk = predict_sequence_topk(model, x, edge_index, batch, topk)

#        print(pred_seq_topk)
        pred_seq_topk = sorted(pred_seq_topk, key=lambda x: x[0], reverse=True)
        
        num = 0
        print("predicted sequence")
        for elem in pred_seq_topk:
            score    = elem[0]
            pred_seq = elem[1]

    #        print("qcid: ", qcid)
            pred_seq_cpu = pred_seq.to('cpu')
            token = ''
            for elem in pred_seq_cpu:
                token += id_to_token[int(elem.item())] + ' '
            print(token + ", " + str(score))
            num += 1
            if num == topk:
                break

        pred_seq = predict_sequence(model, x, edge_index, batch)
#        print("qcid: ", qcid)
        pred_seq_cpu = pred_seq.to('cpu')
        token = ''
        for elem in pred_seq_cpu:
            token += id_to_token[int(elem.item())] + ' '
        print("predicted sequence")
        print(token)
    
        print("Actual sequence:")
        tokens = ''
        for id in sample_graph.y.tolist():
            tokens += (id_to_token[int(id)] + ' ')
        print(tokens)
        print()

        iter += 1
        if iter == 1:
            break
