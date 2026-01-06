from __future__ import annotations
import numpy as np
import networkx as nx
from scipy.linalg import expm


def graph_to_weighted_adjacency(G: nx.DiGraph, n: int, use_abs: bool = True) -> np.ndarray:
    """
    Convert a directed graph into an adjacency matrix A (n x n).

    A[i, j] = weight of edge i -> j
    If use_abs is True, edge weights are converted to absolute values.
    (ignores activation vs repression sign).
    """
    A = np.zeros((n, n), dtype=float)
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        if use_abs:
            w = abs(w)
        A[int(u), int(v)] += w
    return A


def symmetrize(A: np.ndarray) -> np.ndarray:
    """
    Make adjacency matrix symmetric:
    A_sym = (A + A^T) / 2

    Used so that downstream kernels are symmetric and positive semidefinite
    """
    return (A + A.T) / 2.0


def diffusion_node_kernel(A_sym: np.ndarray, beta: float = 1.0, jitter: float = 1e-8) -> np.ndarray:
    """
    Construct a diffusion kernel over genes. (this is the biological prior)

    Steps:
       1. Compute graph Laplacian L = D - A_sym.
       2. Compute diffusion kernel: K = exp(-beta * L)
    
    Interpretations: Genes close in the regualatory network are more similar. 

    Returns: 
        K_gene: (n_genes, n_genes) positive-definite kernel matrix.
    """
    A_sym = np.asarray(A_sym, dtype=float)
    D = np.diag(A_sym.sum(axis=1))
    L = D - A_sym

    # matrix exponential of the laplaian
    K = expm(-beta * L)

    # ensure symmetry
    K = (K + K.T) / 2.0

    # add jitter for numerical stability on the diagonal
    K += np.eye(K.shape[0]) * jitter
    return K


def _k1_set_linear(Xa: np.ndarray, Xb: np.ndarray, K_gene: np.ndarray) -> np.ndarray:
    """
    Linear set kernel between perturbations.

    Each row of X is a multi-hot vector of knocked-out genes

    k1(Xa, Xb) where rows are perturbations:
      k1(x, x') = x^T K_gene x'
    
    Shapes:
        Xa: (na, n_genes)
        Xb: (nb, n_genes)
        K_gene: (n_genes, n_genes)

    Returns:
        K: (na, nb) kernel matrix
    """
    return Xa @ K_gene @ Xb.T


def _rbf_from_Z(Za: np.ndarray, Zb: np.ndarray, length_scale: float) -> np.ndarray:
    """
    RBF kernel between two feature matrices Za (na,d), Zb (nb,d).

    k(z, z') = exp(-||z - z'||^2 / (2 * l^2))
    """
    Za2 = np.sum(Za**2, axis=1, keepdims=True)
    Zb2 = np.sum(Zb**2, axis=1, keepdims=True).T
    d2 = Za2 + Zb2 - 2.0 * (Za @ Zb.T)
    return np.exp(-0.5 * d2 / (length_scale**2 + 1e-12))


def kernel_components(
    Xa: np.ndarray,
    Xb: np.ndarray,
    K_gene: np.ndarray,
    length_scale: float = 1.0,
):
    """
    Compute individual kernel components between perturbations.

    Returns (k1, k2, krbf) as (na, nb) arrays:
      k1 = x^T K_gene x'
      k2 = (k1)^2
      krbf = RBF( K_gene x , K_gene x' )
    """
    k1 = _k1_set_linear(Xa, Xb, K_gene)
    k2 = k1**2

    Za = (K_gene @ Xa.T).T
    Zb = (K_gene @ Xb.T).T
    kr = _rbf_from_Z(Za, Zb, length_scale)
    return k1, k2, kr


def combined_kernel(
    Xa: np.ndarray,
    Xb: np.ndarray,
    K_gene: np.ndarray,
    a1: float = 1.0,
    a2: float = 0.5,
    a3: float = 0.2,
    length_scale: float = 1.0,
):
    """
    Final perturbation kernel as weighted sum of components:

    K = a1^2*k1 + a2^2*k2 + a3^2*krbf

    a1, a2, a3 control relative importance of:
    - linear effects
    - interaction effects
    - smooth nonlinear similarity
    """
    k1, k2, kr = kernel_components(Xa, Xb, K_gene, length_scale=length_scale)
    K = (a1**2) * k1 + (a2**2) * k2 + (a3**2) * kr
    return K


def combined_kernel_diag(
    X: np.ndarray,
    K_gene: np.ndarray,
    a1: float = 1.0,
    a2: float = 0.5,
    a3: float = 0.2,
):
    """
    Compute diag(K(X,X)) without building full matrix.

    Used for GP predictive variance

    For components:
      diag(k1) = x^T K_gene x
      diag(k2) = diag(k1)^2
      diag(krbf) = 1
    """
    XK = X @ K_gene
    d1 = np.sum(XK * X, axis=1)
    d2 = d1**2
    dr = np.ones_like(d1)
    return (a1**2) * d1 + (a2**2) * d2 + (a3**2) * dr
