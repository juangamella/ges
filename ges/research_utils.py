# Copyright 2020 Juan Luis Gamella Martin

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
"""

import numpy as np
import itertools
import networkx as nx
import causaldag as cd # Only used to enumerate MEC
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# TODO: Check that the graph construction is consistent for all functions!!!! i.e. A[i,j] = 1 => i -> j

#---------------------------------------------------------------------
# 

def subsets(S):
    """
    Return an iterator with all possible subsets of the set S.

    Parameters
    ----------
    S : set
        a given set

    Returns
    -------
    subsets : iterable
        an iterable over the subsets of S, in increasing size

    """
    subsets = []
    for r in range(len(S)+1):
        subsets += [set(ss) for ss in itertools.combinations(S, r)]
    return subsets

def member(L, A):
    """Return the index of the first appearance of array A in list
    L. Returns None if A is not in L.
    """
    for i,B in enumerate(L):
        if (A==B).all():
            return i
    return None

def moral_graph(B):
    """Return the moral graph of DAG B"""
    A = (B != 0).astype(int)
    moral = skeleton(A)
    for (i,_,j) in vstructures(A):
        moral[i,j] = 1
        moral[j,i] = 1
    return moral

def all_dags(A, cut=None, debug=False):
    """Return all possible DAGs with the same moral graph as A"""
    moral = moral_graph(A)
    x,y = np.where(moral != 0)
    edge_candidates = list(zip(x,y))

    # All orderings
    p = len(A)
    orders = list(itertools.permutations(range(p)))
    orders = orders if cut is None else orders[:cut]
    
    # All possible DAGs
    dags = [A]
    for order in orders:
        A = np.zeros_like(moral)
        for (fro,to) in edge_candidates:
            if order[fro] > order[to]:
                A[fro,to] = 1
        #assert(is_dag(A))
        dags.append(A)
    dags = np.array(dags)
    return np.unique(dags, axis=0)

def is_dag(B):
    """Returns True if B is the connectivity matrix of a DAG, False
    otherwise"""
    # Note networkx takes B[i,j] = 1 => i -> j
    G = nx.from_numpy_matrix(B, create_using = nx.DiGraph)
    return nx.is_directed_acyclic_graph(G)

def topological_ordering(A):
    """Return a topological ordering for the DAG with adjacency matrix A"""
    # Note networkx takes A[i,j] = 1 => i -> j
    G = nx.from_numpy_matrix(A, create_using = nx.DiGraph)
    return list(nx.algorithms.dag.topological_sort(G))

def scores(true_B, B):
    """
    Given the true connectivity and an estimate, return the precision and recall
    TODO: Test
    """
    true_A = true_B != 0
    A = B != 0
    true_positives = np.logical_and(true_A, A).sum()
    precision = true_positives / A.sum()
    recall = true_positives / true_A.sum()
    return precision, recall

def is_diagonal(M, tol=1e-1):
    """Returns true if a matrix is diagonal, false otherwise
    TODO: Test
    """
    p = len(M)
    M = abs(M)
    M /= np.max(M)
    off_diag = np.logical_not(np.eye(p))
    return not (M[off_diag] > tol).any()

def non_canonical_rows(M, tol=1e-1):
    """Returns the rows i in M which are not the ith canonical unit vector
    TODO: Test"""
    p = len(M)
    M = abs(M)
    M /= np.max(M)
    M[range(p), range(p)] = 0
    return np.where(M > tol)[0]

def is_member_close(L, A, atol=1e-8, rtol=1e-5):
    """Returns True if A is close to a member of L, up to the given
    tolerance. Returns false otherwise"""
    if len(L) == 0:
        return False
    one_hot = members_close(L, A, atol=atol, rtol=rtol)
    return one_hot.any()

def members_close(L, A, atol=1e-8, rtol=1e-5):
    """Returns the location of the members of L which are close to A up to a tolerance"""
    if len(L) == 0:
        return np.array([])
    L = np.array(L)
    axis = tuple(range(1,len(L.shape)))
    one_hot = (abs(L - A) <= atol + rtol * abs(A)).all(axis=axis)
    return one_hot.astype(int)

def unique_close(L, atol=1e-8, rtol=1e-5):
    """Returns the unique arrays of L up to a tolerance. Also returns
    the counts of each element and the inverse (see np.unique)"""
    if len(L) == 0:
        return [], [], []
    L = np.array(L)
    axis = tuple(range(1,len(L.shape)))
    unique = []
    k_hot = np.ones(len(L)) * -1
    counts = []
    indices = list(range(len(L)))
    i = 0
    while len(indices) > 0:
        A = L[indices[0]]
        one_hot = (abs(L - A) <= atol + rtol * abs(A)).all(axis=axis)
        counts.append(one_hot.sum())
        unique.append(A)
        k_hot[one_hot] = i
        [indices.remove(j) for j in np.where(one_hot)[0]]
        i += 1
    return np.array(unique), np.array(counts), k_hot

def compute_mec(A):
    MEC = []
    dag = cd.DAG.from_amat(A.T)
    for dag in dag.cpdag().all_dags():
        MEC.append(build_adjacency(len(A), dag).T)
    return MEC

def build_adjacency(p, edges):
    """Given causaldag's representation of a DAG, transform into a
    adjacency matrix"""
    A = np.zeros((p,p))
    for (i,j) in edges:
        A[i,j] = 1
    return A

def plot_density(X):
    #print(X.min(), X.max(), round(len(X)*1.5))
    xs = np.linspace(X.min(), X.max(), round(len(X)*1.5))
    density = gaussian_kde(X)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    plt.plot(xs,density(xs))

# Very fast way to generate a cartesian product of input arrays
def cartesian(arrays, out=None, dtype=np.byte):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    #dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def matrix_block(M, rows, cols):
    """
    Select a block of a matrix given by the row and column indices
    """
    if not rows or not cols:
        return np.array([[]])
    else:
        (n,m) = M.shape
        idx_rows = np.tile(np.array([rows]).T,len(cols)).flatten()
        idx_cols = np.tile(cols, (len(rows),1)).flatten()
        return M[idx_rows, idx_cols].reshape(len(rows), len(cols))

# Graph definitions for PDAGS
def na(y,x,A):
    """All neighbors of y which are adjacent to x in A"""
    return neighbors(y,A) & adj(x,A)

def neighbors(i,A):
    """The neighbors of i in A, i.e. all nodes connected to i by an
    undirected edge"""
    return set(np.where(np.logical_and(A[i,:] != 0, A[:,i] != 0))[0])
    
def adj(i, A):
    """The adjacent nodes of i in A, i.e. all nodes connected by a
    directed or undirected edge"""
    return set(np.where(np.logical_or(A[i,:] != 0, A[:,i] != 0))[0])

def pa(i, A):
    """The parents of i in A"""
    return set(np.where(np.logical_and(A[:,i] != 0, A[i,:] == 0))[0])

def ch(i, A):
    """The children of i in A"""
    return set(np.where(np.logical_and(A[i,:] != 0, A[:,i] == 0))[0])

def is_clique(S, A):
    """Check if the subgraph of A induced by nodes S is a clique"""
    S = list(S)
    subgraph = A[S,:][:,S]
    subgraph = skeleton(subgraph) # drop edge orientations
    no_edges = np.sum(subgraph != 0)
    n = len(S)
    return no_edges == n * (n-1)

def semi_directed_paths(i,j,A):
    """Return all semi-directed paths from i to j in A"""
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    return list(nx.algorithms.all_simple_paths(G,i,j))

def separates(S,A,B,G):
    """Returns true if the set S separates A from B in G, i.e. if all
    paths in G from nodes in A to nodes in B contain a node in
    S. Exception is raised if S,A and B are not pairwise disjoint.

    Parameters
    ----------
    S : set of ints
        a set of nodes in G
    A : set of ints
        a set of nodes in G
    B : set of ints
        a set of nodes in G
    G : np.array
        the adjacency matrix of the graph, where G[i,j] != 0 => i -> j
        and G[i,j] != 0 and G[j,i] != 0 => i - j.
    
    Returns
    -------
    separated : bool
        if S separates A from B in G

    """
    # Check that sets are pairwise disjoint
    if len(A & B) or len(A & S) or len(B & S):
        raise ValueError("The sets S=%s,A=%s and B=%s are not pairwise disjoint" % (S,A,B))
    for a in A:
        for b in B:
            for path in semi_directed_paths(a,b,G):
                if set(path) & S == set():
                    return False
    return True

def chain_component(i, G):
    """Return all nodes in the connected component of node i after
    dropping all directed edges in G.
    
    Parameters
    ----------
    i : int
        the node's index
    G : np.array
        the adjacency matrix of the graph, where G[i,j] != 0 => i -> j
        and G[i,j] != 0 and G[j,i] != 0 => i - j

    Returns
    -------
    visited : set of ints
        the nodes in the chain component of i

    """
    A = only_undirected(G)
    visited = set()
    to_visit = {i}
    # NOTE: Using a breadth-first search
    while len(to_visit) > 0:
        for j in to_visit:
            visited.add(j)
            to_visit = (to_visit | neighbors(j, A)) - visited
    return visited

def induced_subgraph(S, G):
    """Remove all edges which are not between nodes in S.

    Parameters
    ----------
    S : set of ints
        a set of node indices
    G : np.array
        the adjacency matrix of the graph, where G[i,j] != 0 => i -> j
        and G[i,j] != 0 and G[j,i] != 0 => i - j

    Returns
    -------
    subgraph : np.array
       the adjacency matrix of the resulting graph where all edges
       between nodes not in S are removed. Note that this is not
       really the subgraph, as the nodes not in S still appear as
       disconnected nodes.
    """
    mask = np.zeros_like(G, dtype=np.bool)
    mask[list(S), :] = True
    mask = np.logical_and(mask, mask.T)
    subgraph = np.zeros_like(G)
    subgraph[mask] = G[mask]
    return subgraph

def vstructures(A):

    """
    Return the v-structures of a DAG or PDAG, given its adjacency matrix.

    Parameters
    ----------
    A : np.array
        The adjacency of the (P)DAG, where A[i,j] != 0 => i->j

    Returns
    -------
    vstructs : set()
        the set of v-structures, where every v-structure is a three
        element tuple, e.g. (i,j,k) represents the v-structure
        i -> j <- k, where i < j for consistency.
    
    """
    # Construct the graph with only the directed edges
    dir_A = only_directed(A)
    # Search for colliders in the graph with only directed edges
    colliders = np.where((dir_A != 0).sum(axis=0) > 1)[0]
    # For each collider, and all pairs of parents, check if the
    # parents are adjacent in A
    vstructs = []
    for c in colliders:
        for (i,j) in itertools.combinations(pa(c,A), 2):
            if A[i,j] == 0 and A[j,i] == 0:
                # Ordering might be defensive here, as
                # itertools.combinations already returns ordered
                # tuples; motivation is to not depend on their feature
                vstruct = (i,c,j) if i < j else (j,c,i)
                vstructs.append(vstruct)
    return set(vstructs)
    
def only_directed(P):
    """
    Return the graph with the same nodes as P and only its directed edges.

    Parameters
    ----------
    P : np.array
        adjacency matrix of a graph

    Returns
    -------
    G : np.array
        adjacency matrix of the graph with the same nodes as P and
        only its directed edges

    """
    mask = np.logical_and(P != 0, P.T == 0)
    G = np.zeros_like(P)
    # set to the same values in case P is a weight matrix and there is
    # interest in maintaining the weights
    G[mask] = P[mask]
    return G

def only_undirected(P):
    """
    Return the graph with the same nodes as P and only its undirected edges.

    Parameters
    ----------
    P : np.array
        adjacency matrix of a graph

    Returns
    -------
    G : np.array
        adjacency matrix of the graph with the same nodes as P and
        only its undirected edges

    """
    mask = np.logical_and(P != 0, P.T != 0)
    G = np.zeros_like(P)
    # set to the same values in case P is a weight matrix and there is
    # interest in maintaining the weights
    G[mask] = P[mask]
    return G

def skeleton(A):
    """Return the skeleton of a given graph.

    Parameters
    ----------
    A : np.array
        adjacency matrix of a graph

    Returns
    -------
    S : np.array
        adjacency matrix of the skeleton, i.e. the graph resulting
        from dropping all edge orientations

    """
    return ((A + A.T) != 0).astype(int)
    
def is_consistent_extension(G,P,debug=False):
    """Returns True if the DAG G is a consistent extension of the PDAG
    P. Will raise a ValueError exception if the graph G is not a DAG
    (i.e. cycles or undirected edges).

    Parameters
    ----------
    G : np.array
        the adjacency matrix of DAG
    P : np.array
        the adjacency matrix of PDAG
    debug : bool
        if debugging traces should be outputted

    Returns
    -------
    consistent : bool
        True if G is a consistent extension of P (see below)

    """
    # G is a consistent extension of P iff
    #   0. it is a DAG
    #   1. they have the same v-structures,
    #   2. they have the same skeleton, and
    #   3. every directed edge in P has the same direction in G
    if not is_dag(G):
        raise ValueError("G is not a DAG")
    # v-structures
    same_vstructures = vstructures(P) == vstructures(G)
    # skeleton
    same_skeleton = (skeleton(P) == skeleton(G)).all()
    # same orientation
    directed_P = only_directed(P)
    same_orientation = G[directed_P != 0].all() # No need to check
                                                # transpose as G is
                                                # guaranteed to have
                                                # no undirected edges
    if debug:
        print("v-structures (%s) (P,G): " % same_vstructures,  vstructures(P), vstructures(G))
        print("skeleton (%s) (P,G): " % same_skeleton, skeleton(P), skeleton(G))
        print("orientation (%s) (P,G): " % same_orientation, P, G)
    return same_vstructures and same_orientation and same_skeleton

def sort(L, order=None):
    """Sort the elements in an iterable according to its 'sorted'
    function, or according to a given order: i will precede j if i precedes
    j in the order.

    Parameters
    ----------
    L : iterable
        the iterable to be sorted
    order : iterable or None
        a given ordering. In the sorted result, i will precede j if i
        precedes j in order. If None, i will precede j if i < j

    Returns
    -------
    ordered : list
        a list containing the elements of L, sorted from lesser to
        greater or according to the given order

    """
    L = list(L)
    if order is None:
        return sorted(L)
    else:
        order = list(order)
        pos = np.zeros(len(order), dtype=int)
        pos[order] = range(len(order))
        positions = [pos[l] for l in L]
        return [tup[1] for tup in sorted(zip(positions, L))]
        
