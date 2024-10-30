## Import Section
from sklearn.neighbors import KDTree
from scipy.sparse import *
import numpy as np


def get_top_n_alignments(X,Y,n):
    '''
    Function to produce a KD-Tree and query in order to get the best alignments

    input params: 
    X: Embedding to be aligned
    Y: Embedding to which the alignment should be conducted
    n: Number of top alignments

    return params: 
    d: Array including the distance to each of the nearest nodes
    i: List of indices of the nearest nodes
    '''
    # Create an embedding of Y in a KD-Tree, euclidean distance is used in accordance with the paper
    kd_tree = KDTree(Y, metric = "euclidean")

    # Query that tree to get the alignment distances and indices
    d,i = kd_tree.query(X, k = n)

    return d,i


def get_similarity_matrix(X,Y,n): 
    '''
    Function to calculate tge similarity matrix as per the REGAL Paper
    input params: 
    X: Embedding to be aligned
    Y: Embedding to which the alignment should be conducted
    n: Number of top alignments

    return params: 

    '''
    # Get the top n alignments
    distances,indices = get_top_n_alignments(X,Y,n)

    # Initalize the DOK Matrix
    sparse_align_matrix = dok_matrix((X.shape[0], Y.shape[0]))
    # Loop through all elements to be embedded
    for i in range(X.shape[0]):
        for j in range(n):
            row_index = i
            col_index = indices[i, j]
            # Populate a DOK matrix with similarity scores as defined   
            sparse_align_matrix[row_index, col_index] = np.exp(-distances[i, j])

    return sparse_align_matrix.tocsr()


def split_embeddings(embedding_matrix):
    '''
    Helper Function to split the embeddings if necessary

    input params: 
    embedding_matrix: Matrix including the combined embeddings
    return params: 
    X,Y: Split embeddings
    '''

    split_index= int(embedding_matrix.shape[0] / 2)
    X = embedding_matrix[:split_index]
    Y = embedding_matrix[split_index:]

    return X, Y

