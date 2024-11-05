import numpy as np

def get_number_of_landmarks(graph, rep_method, k=10):
    """
    Given a graph, computes the number of landmark nodes.

    Inputs:
        - graph: A Graph object.
        - rep_method: A Rep_Method object.
        - k: A user-defined parameter that adjusts how many landmark nodes there are.

    Outputs:
    """
    rep_method.p = min(graph.N, int(k*np.log2(graph.N)))

def get_random_landmarks(graph, rep_method):
    """
    Given a graph and a rep_method, returns a random sample of rep_method.p landmark nodes, chosen without replacement.
    """  
    return np.random.choice(graph.N, rep_method.p, replace=False)

def compute_similarity_score(d_u: np.ndarray, d_v: np.ndarray, gamma_s=1, gamma_a=None, f_u=None, f_v=None):
    """
    Helper function. Given two vectors that come from a feature matrix, computes the similarity between them according to equation (1) of the paper.

    Inputs:
        - d_u: First vector for node u.
        - d_v: Second vector for node v.
        - gamma_s: User-defined scalar parameter controlling the effect of the structural identity.
        - gamma_a: User-defined scalar parameter controlling the effect of the attribute identity. TODO: implement.
        - f_u: Attribute vector for node u. TODO: implement.
        - f_v: Attribute vector for node v. TODO: implement.
    """
    # Note: the paper says take norm squared, while the source code they provide says take norm. 
    # Here, I take norm squared to align with the paper.
    return np.exp(-1*gamma_s*np.linalg.norm(d_u - d_v)**2) 

def compute_C_matrix(feature_matrix, landmarks):
    """
    Given a feature matrix and a list of landmark nodes, computes the similarity matrix C between all the n nodes and the p landmark nodes.

    Inputs:
        - feature_matrix: The feature matrix from step 1 of the algorithm.
        - landmarks: The list of landmark nodes of the graph.
    Outputs:
        - C: The n x p similarity matrix.
    """
    C = np.zeros((len(feature_matrix), len(landmarks)))

    for n in range(len(feature_matrix)):
        for j in range(len(landmarks)):
            C[n, j] = compute_similarity_score(d_u=feature_matrix[n], d_v = feature_matrix[landmarks[j]])

    return C

def compute_representation(C, landmarks, n_1):
    """
    Given an n x p similarity matrix C, a list of landmark nodes, and the number of nodes in the first graph, computes the representations of nodes of the two original graphs. See algorithm 2, step 2b, for this function in pseudocode.

    Inputs:
        - C: n x p similarity matrix.
        - landmarks: The list of landmark nodes in the graph.
        - n_1: Number of nodes in the first graph.

    Outputs:
        - Y_twiddle_1: Representations of the nodes of the first graph.
        - Y_twiddle_2: Representations of the nodes of the second graph.
    """
    W = C[landmarks, :] # Select the rows of C that correspond to the landmark nodes
    W_dagger = np.linalg.pinv(W) # Pseudoinverse of landmark-to-landmark matrix
    U, Sigma, V = np.linalg.svd(W_dagger) # Singular value decomposition
    Sigma = np.diag(Sigma) # Diagonalization of resulting vector
    Y_twiddle = C @ U @ np.sqrt(Sigma) # Matrix multiplication
    Y_twiddle = Y_twiddle / np.linalg.norm(Y_twiddle, axis=1).reshape(Y_twiddle.shape[0], 1) # Normalization of rows
    Y_twiddle_1, Y_twiddle_2 = Y_twiddle[:n_1, :], Y_twiddle[n_1:, :] # Split representations for nodes in G_1, G_2

    return Y_twiddle_1, Y_twiddle_2