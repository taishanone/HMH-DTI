import numpy as np
import csv
n=1094
m=1556
DTI = np.loadtxt('./data/zheng/mat_drug_protein.txt')
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

np.random.seed(2024)
index_1 = np.loadtxt('divide_result_zheng/index_pos.txt')
index_0 = np.loadtxt('divide_result_zheng/index_neg.txt')
index = np.hstack((index_1, index_0))
train_index = index.flatten()
np.random.shuffle(train_index)
train_lable,train_row_idx,train_column_idx=get_lable(DTI,train_index,n,m)
drug_file = './data/zheng/drug.txt'
drug_smiles_file = './data/zheng/drug_smiles.txt'
protein_file = './data/zheng/protein.txt'
protein_seq_file = './data/zheng/protein_seq.txt'
output_file = './data/zheng/zheng.csv'

def read_file(file_path):
    with open(file_path, 'r') as f:
        return f.readlines()

drug_lines = read_file(drug_file)
drug_smiles_lines = read_file(drug_smiles_file)
protein_lines = read_file(protein_file)
protein_seq_lines = read_file(protein_seq_file)


with open(output_file, 'w',newline='',encoding='utf-8') as output_f:
    csv_writer = csv.writer(output_f)
    csv_writer.writerow(['smiles', 'sequence', 'label'])

    for row_idx, col_idx, label in zip(train_row_idx, train_column_idx, train_lable):

        drug = drug_lines[row_idx].strip()
        drug_smiles = drug_smiles_lines[row_idx].strip()
        protein = protein_lines[col_idx].strip()
        protein_seq = protein_seq_lines[col_idx].strip()
        label_int = int(label)

        csv_writer.writerow([drug_smiles, protein_seq, label_int])
print("数据处理完成，已存入zheng.csv。")