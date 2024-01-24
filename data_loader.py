import json
import torch
from torch_geometric.data import Data

def load_data_from_json(filename, token_to_id = {}, id_to_token = {}, max_node_label = 0, max_edge_label = 0):
    with open(filename, 'r') as f:
        graph_data_list = json.load(f)

    token = '+'
    if token not in token_to_id:
        token_to_id[token] = len(token_to_id)
        id_to_token[token_to_id[token]] = token

    token = '*'
    if token not in token_to_id:
        token_to_id[token] = len(token_to_id)
        id_to_token[token_to_id[token]] = token

    cids = []
    list_graph_node_features = []
    data_list = []
    for graph_data in graph_data_list:
        cid = graph_data["cid: "]
        cids.append(cid)
        
        unique_tokens = set(graph_data["seq"].split())
        for idx, token in enumerate(unique_tokens):
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)
                id_to_token[token_to_id[token]] = token

        # シーケンスを整数のリストに変換
        sequence = []
        tokens = graph_data["seq"].split()
        sequence.append(token_to_id['+'])
        for s in tokens:
            sequence.append(token_to_id[s])
#        sequence.append(token_to_id['*'])

        list_node_features = []
        for i in range(len(graph_data["g_ids"])):
            node_features = []
            for feature in graph_data["g_ids_features"][str(i)]:
                if feature == ' ':
                    continue

                node_features.append(int(feature))

            list_node_features.append(node_features)

        list_graph_node_features.append(list_node_features)

        # エッジを作成
        edge_index = []
        for src, dests in graph_data["g_adj"].items():
            for dest in dests:
                edge_index.append((int(src), int(dest)))
                edge_index.append((int(dest), int(src)))
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        data = Data(x=torch.tensor(node_features, dtype=torch.float), edge_index=edge_index, y=torch.tensor(sequence, dtype=torch.long))

        data_list.append(data)


    # 最大の長さを取得
#    if max_length == 0:
#        max_length = max(len(lst) for sublist in list_graph_node_features for lst in sublist)

    # 最大の要素を取得
#    if max_element == 0:
#        max_element = max(max(lst) for sublist in list_graph_node_features for lst in sublist if lst)  # 空のリストを無視

    if max_node_label == 0:
        for graph_node_features in list_graph_node_features:
            for node_features in graph_node_features:
                if len(node_features) > 0:
                    if max_node_label < node_features[0]:
                        max_node_label = node_features[0]

    if max_edge_label == 0:
        for graph_node_features in list_graph_node_features:
            for node_features in graph_node_features:
                for i in range(len(node_features) - 1):
                    if max_edge_label < node_features[i+1]:
                        max_edge_label = node_features[i+1]

    g_id = 0
    for graph_node_features in list_graph_node_features: 
        list_node_feature_vec = []
        for node_features in graph_node_features: #1つのグラフに対するノードの特徴量のリスト#
            node_feature_vec = [0] * (max_node_label + max_edge_label + 1)
            if len(node_features) > 0:
                elem = node_features[0]
                node_feature_vec[elem] = 1
            
                for i in range(len(node_features)-1):
                    elem = node_features[i+1]
                    node_feature_vec[max_node_label + elem] = 1

            list_node_feature_vec.append(node_feature_vec)
        
        data_list[g_id].x = torch.tensor(list_node_feature_vec, dtype=torch.float)
        g_id += 1

    return cids, data_list, token_to_id, id_to_token, max_node_label, max_edge_label
