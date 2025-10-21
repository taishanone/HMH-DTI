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

    drug_drug_interaction = np.loadtxt('./data/mat_drug_drug.txt')
    drug_disease_association = np.loadtxt('./data/mat_drug_disease.txt')
    drug_sideeffect_association = np.loadtxt('./data/mat_drug_se.txt')
    Similarity_Matrix_Drugs = np.loadtxt('./data/Similarity_Matrix_Drugs.txt')

    drug_disease_association_similarity = Jaccard_similarity(drug_disease_association)
    np.savetxt("./comprehensive_similar_matrix/SDSI.txt", drug_disease_association_similarity)
    drug_drug_interaction_similarity = Jaccard_similarity(drug_drug_interaction)
    np.savetxt("./comprehensive_similar_matrix/SDDI.txt", drug_drug_interaction_similarity)
    drug_sideeffect_association_similarity = Jaccard_similarity(drug_sideeffect_association)
    np.savetxt("./comprehensive_similar_matrix/SDAI.txt", drug_sideeffect_association_similarity)

    data = np.loadtxt("./comprehensive_similar_matrix/SDSI.txt")
    data1 = np.loadtxt("./comprehensive_similar_matrix/SDDI.txt")
    data2 = np.loadtxt("./comprehensive_similar_matrix/SDAI.txt")
    Drug_comprehensive_similarity_matrix = (data + data1 + data2 + Similarity_Matrix_Drugs) / 5
    np.savetxt("./data/Drug_comprehensive_similarity_matrix.txt", Drug_comprehensive_similarity_matrix)

    target_disease_association = np.loadtxt('./data/mat_protein_disease.txt')
    target_target_interaction = np.loadtxt('./data/mat_protein_protein.txt')
    Similarity_Matrix_Proteins = np.loadtxt('./data/Similarity_Matrix_Proteins.txt')

    target_disease_association_similarity = Jaccard_similarity(target_disease_association)
    np.savetxt("./comprehensive_similar_matrix/STSI.txt", target_disease_association_similarity)
    target_target_interaction_similarity = Jaccard_similarity(target_target_interaction)
    np.savetxt("./comprehensive_similar_matrix/STTI.txt", target_target_interaction_similarity)

    data3 = np.loadtxt("./comprehensive_similar_matrix/STSI.txt")
    data4 = np.loadtxt("./comprehensive_similar_matrix/STTI.txt")
    Protein_comprehensive_similarity_matrix = (data3 + data4 + Similarity_Matrix_Proteins) / 4
    np.savetxt("./data/Protein_comprehensive_similarity_matrix.txt", Protein_comprehensive_similarity_matrix)

