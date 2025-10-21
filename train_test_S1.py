import os, json, torch
import numpy as np
from collections import OrderedDict
import scipy.sparse as sp
from torch_geometric import data as DATA
from model import ConvNet_or_GAT, Predictor,wo_muti_modal_ConvNet_or_GAT,wo_Hete_net_ConvNet_or_GAT,join_model_hete_ConvNet_or_GAT,target_rep_ESM_ConvNet_or_GAT
from metrics import model_evaluate,evaluate
from GraphInput import getDrugMolGraph,getDrugsmiles,getTargetESM2,getTargetmolGraph,getTatgetSeq

from utils import argparser, DTADataset, GraphDataset, collate, getLinkEmbeddings, predicting, read_data, train,GraphDataset_Drug_target
import argparse
import random
import copy
import numpy as np
import math
from utils1.utils import set_seed, tab_printer
device = torch.device("cuda:0")
DTI = np.loadtxt('./data/zheng/mat_drug_protein.txt')
n = DTI.shape[0]
m = DTI.shape[1]

def get_lable(label,index,drug_num, target_num):
    row=[]
    column=[]
    output = []
    for i in range(index.shape[0]):
        drug = int(index[i] / target_num)
        target = int(index[i] % target_num)
        row.append(drug)
        column.append(target)
        output.append(label[drug, target])
    return np.array(output),np.array(row),np.array(column)

def creat_train_test_hgraph(fold):
    index_1 = np.loadtxt('divide_result_zheng/index_pos.txt')
    index_0 = np.loadtxt('divide_result_zheng/index_neg.txt')
    index = np.hstack((index_1, index_0))
    idx = copy.deepcopy(index)
    test_index = copy.deepcopy(idx[fold])
    idx = np.delete(idx, fold, axis=0)
    train_index = idx.flatten()
    insersection = np.intersect1d(test_index, train_index)
    if insersection.size > 0:
        raise ValueError("There is an intersection between the test set and the training set")
    np.random.shuffle(train_index)
    train_lable,train_row_idx,train_column_idx=get_lable(DTI,train_index,n,m)
    test_lable,test_row_idx,test_column_idx=get_lable(DTI,test_index,n,m)
    tem_test_row_idx=test_row_idx
    tem_test_column_idx=[]
    for i in range(len(test_column_idx)):
        tem_test_column_idx.append(test_column_idx[i]+708)

    train_dataset = DTADataset(drug_ids=train_row_idx, target_ids=train_column_idx, y=train_lable)
    test_dataset =  DTADataset(drug_ids=test_row_idx, target_ids=test_column_idx, y=test_lable)

    A = np.loadtxt('./network_data_zheng/A' + str(fold) + '.txt')
    edge_index = sp.coo_matrix(A)
    edge_index = np.vstack((edge_index.row, edge_index.col)).T
    edge_index = torch.LongTensor(edge_index)

    x = np.loadtxt('./network_data_zheng/X.txt')
    x = torch.Tensor(x)
    x = x.float()
    Hgraph = DATA.Data(x=x,adj=torch.Tensor(A),edge_index=edge_index)
    Hgraph.__setitem__("num_node1s", n)
    Hgraph.__setitem__("num_node2s", m)

    np.random.seed(2024)

    #获得正样本
    edge_index1=sp.coo_matrix(A)
    positive_rows = edge_index1.row
    positive_cols = edge_index1.col
    positive_labels = np.ones(len(positive_rows))
    #获得负样本
    np.fill_diagonal(A, 1)
    A[tem_test_row_idx,tem_test_column_idx] = 1
    A[tem_test_column_idx,tem_test_row_idx] = 1
    zero_indices = np.nonzero(A == 0)
    num_positive_samples = len(positive_rows)
    sample_indices = np.random.choice(len(zero_indices[0]), num_positive_samples, replace=False)
    negative_rows = zero_indices[0][sample_indices]
    negative_cols = zero_indices[1][sample_indices]
    negative_labels = np.zeros(len(negative_rows))

    #合并
    combined_rows = np.concatenate((positive_rows, negative_rows))
    combined_cols = np.concatenate((positive_cols, negative_cols))
    combined_labels = np.concatenate((positive_labels, negative_labels))
    train_dataset= DTADataset(drug_ids=combined_rows,target_ids=combined_cols,y=combined_labels)

    return train_dataset,test_dataset,Hgraph

def train_test():

    FLAGS = argparser()

    dataset = FLAGS.dataset
    cuda_name = f'cuda:{FLAGS.cuda_id}'
    TRAIN_BATCH_SIZE = FLAGS.batch_size
    TEST_BATCH_SIZE = FLAGS.batch_size
    NUM_EPOCHS = FLAGS.num_epochs
    LR = FLAGS.lr
    Architecture = [ConvNet_or_GAT, wo_muti_modal_ConvNet_or_GAT, wo_Hete_net_ConvNet_or_GAT,join_model_hete_ConvNet_or_GAT, target_rep_ESM_ConvNet_or_GAT][FLAGS.model]
    model_name = Architecture.__name__
    fold = FLAGS.fold
    if not FLAGS.weighted:
        model_name += "-noweight"
    if fold != -100:
        model_name += f"-{fold}"

    print("Dataset:", dataset)
    print("Cuda name:", cuda_name)
    print("Epochs:", NUM_EPOCHS)
    print("Learning rate:", LR)
    print("Model name:", model_name)
    print("Train and test") if fold == -100 else print("Fold of 5-CV:", fold)
    
    if os.path.exists(f"models/architecture/{dataset}/S1/cross_validation/") is False:
        os.makedirs(f"models/architecture/{dataset}/S1/cross_validation/")
    if os.path.exists(f"models/predictor/{dataset}/S1/cross_validation/") is False:
        os.makedirs(f"models/predictor/{dataset}/S1/cross_validation/")
    if os.path.exists(f"models/architecture/{dataset}/S1/test/") is False:
        os.makedirs(f"models/architecture/{dataset}/S1/test/")
    if os.path.exists(f"models/predictor/{dataset}/S1/test/") is False:
        os.makedirs(f"models/predictor/{dataset}/S1/test/")

    print("create dataset ...")
    train_data, test_data, Hgraph_my = creat_train_test_hgraph(fold)

    print("create train_loader and test_loader ...")

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=512, shuffle=True, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=77, shuffle=True, collate_fn=collate)

    print("create drug_graphs_dict and target_graphs_dict ...")

    drug_graphs_dict_my = getDrugMolGraph(f'data/{dataset}/drug_smiles.txt')
    drug_smiles_my=getDrugsmiles(f'data/{dataset}/drug_smiles.txt')




    target_ESM_2_my = getTargetESM2(f'data/{dataset}/final_embedding')
    target_graphs_dict_my=getTargetmolGraph(f'data/{dataset}/protein.txt',f'data/{dataset}/protein_seq.txt',dataset)

    print("create drug_graphs_DataLoader and target_graphs_DataLoader ...")

    drug_graphs_Data_my = GraphDataset_Drug_target(graphs_dict=drug_graphs_dict_my, dttype="drug")


    drug_graphs_DataLoader_my = torch.utils.data.DataLoader(drug_graphs_Data_my, shuffle=False, collate_fn=collate, batch_size=Hgraph_my.num_node1s)



    target_graphs_Data_my=GraphDataset_Drug_target(graphs_dict=target_graphs_dict_my,dttype="target")
    target_graphs_DataLoader_my=torch.utils.data.DataLoader(target_graphs_Data_my, shuffle=False, collate_fn=collate, batch_size=Hgraph_my.num_node2s)

    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    architecture = Architecture(ag_init_dim=512, HGraph_dropout_rate=FLAGS.dropedge_rate, skip=FLAGS.skip)
    architecture.to(device)

    predictor = Predictor(embedding_dim=architecture.output_dim)
    predictor.to(device)

    if fold != -100:
        best_result = [0]
    print("start training ...")
    flag=0

    flag_file_result="./best_result.txt"
    flag_file_num="./best_num.txt"
    set_seed(FLAGS.seed)
    for epoch in range(NUM_EPOCHS):
        train(architecture, predictor, device, train_loader, drug_graphs_DataLoader_my, target_graphs_DataLoader_my, LR, epoch + 1, TRAIN_BATCH_SIZE, Hgraph_my,drug_smiles_my,target_ESM_2_my)
        G, P = predicting(architecture, predictor, device, test_loader, drug_graphs_DataLoader_my, target_graphs_DataLoader_my, Hgraph_my,drug_smiles_my,target_ESM_2_my)

        result=evaluate(G,P)
        if (epoch+1) % 100 == 0:
            print(f"---------result {epoch+1}:",best_result)
            print(flag)

            with open(flag_file_result, "w") as f:
                f.write(f"{best_result}\n")
            with open(flag_file_num, "w") as f:
                f.write(f"{flag}\n")

        if(epoch+1)==NUM_EPOCHS:
            print("The last result:",result)
        if fold != -100 and result[0] > best_result[0]:
            best_result = result
            flag=epoch+1
            checkpoint_dir = f"models/architecture/{dataset}/S1/cross_validation/"
            checkpoint_path = checkpoint_dir + model_name + ".pkl"
            torch.save(architecture.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)
            
            checkpoint_dir = f"models/predictor/{dataset}/S1/cross_validation/"
            checkpoint_path = checkpoint_dir + model_name + ".pkl"
            torch.save(predictor.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

    if fold == -100:
        checkpoint_dir = f"models/architecture/{dataset}/S1/test/"
        checkpoint_path = checkpoint_dir + model_name + ".pkl"
        torch.save(architecture.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

        checkpoint_dir = f"models/predictor/{dataset}/S1/test/"
        checkpoint_path = checkpoint_dir + model_name + ".pkl"
        torch.save(predictor.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

        print('\npredicting for test data')
        G, P = predicting(architecture, predictor, device, test_loader, drug_graphs_DataLoader_my, target_graphs_DataLoader_my, Hgraph_my)
        result = model_evaluate(G, P, dataset)
        print("reslut:", result)
    else:
        print(f"\nbest result for fold {fold} of cross validation:")
        print("reslut:", best_result)


if __name__ == '__main__':
    train_test()
