import numpy as np
from tqdm import tqdm
import time


# Jaccard similarity matrix calculation
def Jaccard_similarity(A):
    B = np.zeros((A.shape[0], A.shape[0]))
    for i in tqdm(range(B.shape[0])):
        for j in range(i + 1, B.shape[1]):
            if np.sum(A[i]) == 0 and np.sum(A[j]) == 0:
                B[i][j] = 0
            else:
                jiaoji = 0
                bingji = 0
                for k in range(A.shape[1]):
                    if A[i][k] == 1 and A[j][k] == 1:
                        jiaoji += 1
                        bingji += 1
                    elif A[i][k] == 1 or A[j][k] == 1:
                        bingji += 1
                B[i][j] = jiaoji / bingji
    row, col = np.diag_indices_from(B)
    B[row, col] = 1
    B += B.T - np.diag(B.diagonal())
    return B


if __name__ == "__main__":

    drug_drug_interaction = np.loadtxt('./data/zheng/mat_drug_drug.txt')
    drug_chemical_association = np.loadtxt('./data/zheng/mat_drug_chemical_substructures.txt')
    drug_sideeffect_association = np.loadtxt('./data/zheng/mat_drug_sideeffects.txt')
    drug_substituent_association=np.loadtxt('./data/zheng/mat_drug_sub_stituent.txt')
    Similarity_Matrix_Drugs = np.loadtxt('./data/zheng/Similarity_Matrix_Drugs.txt')

    drug_drug_interaction_similarity = Jaccard_similarity(drug_drug_interaction)
    np.savetxt("./comprehensive_similar_matrix_zheng/SDDI.txt", drug_drug_interaction_similarity)
    drug_chemical_interaction_similarity = Jaccard_similarity(drug_chemical_association)
    np.savetxt("./comprehensive_similar_matrix_zheng/SDCI.txt", drug_chemical_interaction_similarity)
    drug_sideeffect_association_similarity = Jaccard_similarity(drug_sideeffect_association)
    np.savetxt("./comprehensive_similar_matrix_zheng/SDEI.txt", drug_sideeffect_association_similarity)
    drug_substituent_association_similarity=Jaccard_similarity(drug_substituent_association)
    np.savetxt("./comprehensive_similar_matrix_zheng/SDSI.txt", drug_substituent_association_similarity)

    data = np.loadtxt("./comprehensive_similar_matrix_zheng/SDDI.txt")
    data1 = np.loadtxt("./comprehensive_similar_matrix_zheng/SDCI.txt")
    data2 = np.loadtxt("./comprehensive_similar_matrix_zheng/SDEI.txt")
    data3 = np.loadtxt("./comprehensive_similar_matrix_zheng/SDSI.txt")
    Drug_comprehensive_similarity_matrix = (data + data1 + data2 + data3 + Similarity_Matrix_Drugs) / 5

    target_go_association = np.loadtxt('./data/zheng/mat_target_GO.txt')
    target_target_interaction = np.loadtxt('./data/zheng/mat_protein_protein.txt')
    Similarity_Matrix_Proteins = np.loadtxt('./data/zheng/Similarity_Matrix_Proteins.txt')

    target_go_association_similarity = Jaccard_similarity(target_go_association)
    np.savetxt("./comprehensive_similar_matrix_zheng/STGI.txt", target_go_association_similarity)
    target_target_interaction_similarity = Jaccard_similarity(target_target_interaction)
    np.savetxt("./comprehensive_similar_matrix_zheng/STTI.txt", target_target_interaction_similarity)

    data3 = np.loadtxt("./comprehensive_similar_matrix_zheng/STGI.txt")
    data4 = np.loadtxt("./comprehensive_similar_matrix_zheng/STTI.txt")
    Protein_comprehensive_similarity_matrix = (data3 + data4 + Similarity_Matrix_Proteins) / 3
    np.savetxt("./data/zheng/Protein_comprehensive_similarity_matrix.txt", Protein_comprehensive_similarity_matrix)

