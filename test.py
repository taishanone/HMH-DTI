# author smile
# 2025年03月06日
dataset='zheng'
def target_to_graph(target_key, target_sequence, contact_dir, aln_dir):
    target_size = len(target_sequence)
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    # contact_dir = 'data/' + dataset + '/pconsc4'

    target_feature = target_to_feature(target_key, target_sequence, aln_dir)

    contact_map = np.load(contact_file)
    contact_map += np.matrix(np.eye(target_size))
    index_row, index_col = np.where(contact_map >= 0.5)
    target_edge_index = []
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
    target_edge_index = np.array(target_edge_index)
    return target_size, target_feature, target_edge_index
def getTargetmolGraph(protein_id_path,protein_seq_path,dataset):
    # load contact and aln
    msa_path = 'data/' + dataset + '/aln'
    contac_path = 'data/' + dataset + '/pconsc4'
    # create protein graph
    print("create protein graph ...")
    target_graph=[]
    # i=1 #记得去掉
    with open(protein_id_path,"r") as id_file,open(protein_seq_path,"r") as seq_file:
        for id_line,seq_line in zip(id_file,seq_file):
            # print(i)
            # if i==0:
            #     i+=1
            #     continue
            protein_id = id_line.strip()  # 去掉 ID 的首尾空白符
            sequence = seq_line.strip()  # 去掉序列的首尾空白符
            g=target_to_graph(protein_id,sequence,contac_path,msa_path)#顺序读取记得random
            target_graph.append(g)
            # i+=1
    print('effective protein:', len(target_graph))     #12条异常数据，过长的截断为1500，存在U的去掉，继续生成（恰好作为噪声数据）Zheng数据集9条截断为1500
    if len(target_graph) == 0:
        raise Exception('no protein, run the script for datasets preparation.')

    return target_graph
target_graphs_dict_my=getTargetmolGraph(f'data/{dataset}/protein.txt',f'data/{dataset}/protein_seq.txt',dataset)