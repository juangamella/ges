# Copyright 2021 Juan L Gamella

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
The module containing the implementation of GES, including
the logic for the insert, delete and turn operators.
"""

import numpy as np
import ges.utils as utils

# TODO: Update sempler.utils version of matrix_block
# TODO: ASSUME DATA IS CENTERED
# TODO: Raise exception if x or y are contained in H or T
# TODO: If condition 1 does not pass for an operator, do not check condition 2
# TODO: Update research utils with new changes
# TODO: Rubber duck turning logic and semi_directed_paths
# TODO: Write README & requirements
# TODO: Finish documenting all functions

def fit(cache, A0, phases = ['forward', 'backward', 'turning'], iterate=False, debug=0):
    # Select the desired phases
    if len(phases) == 0:
        raise ValueError("Must specify at least one phase")
    # GES procedure
    total_score = 0
    A, score_change = A0, np.Inf
    for phase in phases:
        if phase == 'forward':
            fun = forward_step
        elif phase == 'backward':
            fun = backward_step
        elif phase == 'turning':
            fun = turning_step
        else:
            raise ValueError('Invalid phase "%s" specified' % phase)
        print("\nGES %s phase start" % phase) if debug else None
        print("-------------------------") if debug else None
        while True:
            score_change, new_A = fun(A, cache, max(0,debug-1))
            if  score_change > 0:
                A = utils.pdag_to_cpdag(new_A)
                total_score += score_change
            else:
                break
        print("-----------------------") if debug else None
        print("GES %s phase end" % phase) if debug else None
        print("Total score: %0.4f" % total_score) if debug else None
        [print(row) for row in A] if debug else None
    return A, total_score
    
def forward_step(A, cache, debug=0):
    # Construct edge candidates (i.e. edges between non-adjacent nodes)
    fro,to = np.where((A + A.T + np.eye(len(A))) == 0)
    edge_candidates = list(zip(fro,to))
    # For each edge, enumerate and score all valid operators
    valid_operators = []
    print("  %d candidate edges" % len(edge_candidates)) if debug>1 else None
    for (x,y) in edge_candidates:
        valid_operators += score_valid_insert_operators(x,y,A,cache, debug = max(0,debug-1))
    # Pick the edge/operator with the highest score
    if len(valid_operators) == 0:
        print("  No valid insert operators remain") if debug else None
        return 0, A
    else:
        scores = [op[0] for op in valid_operators]
        score, new_A, x, y, T = valid_operators[np.argmax(scores)]
        print("  Best operator: insert(%d, %d, %s) -> (%0.4f)" % (x, y, T, score)) if debug else None
        return score, new_A

def backward_step(A, cache, debug=0):
    # Construct edge candidates:
    #   - directed edges
    #   - undirected edges, counted only once
    fro,to = np.where(utils.only_directed(A))
    directed_edges = zip(fro,to)
    fro,to = np.where(utils.only_undirected(A))
    undirected_edges = filter(lambda e: e[0] > e[1], zip(fro,to))#zip(fro,to)
    edge_candidates = list(directed_edges) + list(undirected_edges)
    assert len(edge_candidates) == utils.skeleton(A).sum() / 2
    # For each edge, enumerate and score all valid operators
    valid_operators = []
    print("  %d candidate edges" % len(edge_candidates)) if debug>1 else None
    for (x,y) in edge_candidates:
        valid_operators += score_valid_delete_operators(x,y,A,cache, debug = max(0,debug-1))
    # Pick the edge/operator with the highest score
    if len(valid_operators) == 0:
        print("  No valid delete operators remain") if debug else None
        return 0, A
    else:
        scores = [op[0] for op in valid_operators]
        score, new_A, x, y, H = valid_operators[np.argmax(scores)]
        print("  Best operator: delete(%d, %d, %s) -> (%0.4f)" % (x, y, H, score)) if debug else None
        return score, new_A

def turning_step(A, cache, debug=0):
    # Construct edge candidates:
    #   - directed edges, reversed
    #   - undirected edges
    fro,to = np.where(A != 0)
    edge_candidates = list(zip(to,fro))
   # For each edge, enumerate and score all valid operators
    valid_operators = []
    print("  %d candidate edges" % len(edge_candidates)) if debug>1 else None
    for (x,y) in edge_candidates:
        valid_operators += score_valid_turn_operators(x,y,A,cache, debug = max(0,debug-1))
    # Pick the edge/operator with the highest score
    if len(valid_operators) == 0:
        print("  No valid turn operators remain") if debug else None
        return 0, A
    else:
        scores = [op[0] for op in valid_operators]
        score, new_A, x, y, C = valid_operators[np.argmax(scores)]
        print("  Best operator: turn(%d, %d, %s) -> (%0.4f)" % (x, y, C, score)) if debug else None
        return score, new_A

# --------------------------------------------------------------------
# Delete operator
#    1. definition in function delete
#    2. enumeration logic (to enumerate and score only valid
#    operators) function in valid_delete_operators

def delete(x,y,H,A):
    """
    Applies the delete operator:
      1) deletes the edge x -> y or x - y
      2) for every node h in H
           * orients the edge y -> h
           * if the edge with x is undirected, orients it as x -> h

    Note that H must be a subset of the neighbors of y which are
    adjacent to x. A ValueError exception is thrown otherwise.

    Parameters
    ----------
    x : int
        the "origin" node (i.e. x -> y or x - y)
    y : int
        the "target" node
    H : iterable of ints
        a subset of the neighbors of y which are adjacent to x
    A : np.array
        the current adjacency matrix
    
    Returns
    -------
    new_A : np.array
        the adjacency matrix resulting from applying the operator

    """
    H = set(H)
    # Check inputs
    if A[x,y] == 0:
        raise ValueError("There is no (un)directed edge from x=%d to y=%d" % (x,y))
    # neighbors of y which are adjacent to x
    na_yx = utils.na(y,x,A)
    if not H <= na_yx:
        raise ValueError("The given set H is not valid, H=%s is not a subset of NA_yx=%s" % (H, na_yx))
    # Apply operator
    new_A = A.copy()
    # delete the edge between x and y
    new_A[x,y], new_A[y,x] = 0, 0
    # orient the undirected edges between y and H towards H
    new_A[list(H),y] = 0
    # orient any undirected edges between x and H towards H
    n_x = utils.neighbors(x,A)
    new_A[list(H & n_x), x] = 0
    return new_A

def score_valid_delete_operators(x,y,A,cache,debug=0):
    """Generate and score all valid delete(x,y,H) operators involving the edge
    x -> y or x - y, and all possible subsets H of neighbors of y which
    are adjacent to x.
   
    Parameters
    ----------
    x : int
        the "origin" node (i.e. x -> y or x - y)
    y : int
        the "target" node
    A : np.array
        the current adjacency matrix
    cache : instance of ges.scores.DecomposableScore
        the score cache to compute the score of the
        operators that are valid
    debug : bool or string
        if debug traces should be printed (True/False). If a non-empty
        string is passed, traces are printed with the given string as
        prefix (useful for indenting the prints from a calling
        function)

    Returns
    -------
    valid_operators : list of tuples
        a list of tubles, each containing a valid operator, its score
        and the resulting connectivity matrix

    """
    # Check inputs
    if A[x,y] == 0:
        raise ValueError("There is no (un)directed edge from x=%d to y=%d" % (x,y))
    # One-hot encode all subsets of H0, plus one column to mark if
    # they have already passed the validity condition
    na_yx = utils.na(y,x,A)
    H0 = sorted(na_yx)
    p = len(A)
    if len(H0) == 0:
        subsets = np.zeros((1,(p+1)), dtype=np.bool)
    else:
        subsets = np.zeros((2**len(H0), (p+1)), dtype=np.bool)
        subsets[:, H0] = utils.cartesian([np.array([False, True])] * len(H0), dtype=np.bool)
    valid_operators = []
    print("    delete(%d,%d) H0=" % (x,y),set(H0)) if debug>1 else None
    while len(subsets) > 0:
        print("      len(subsets)=%d, len(valid_operators)=%d" % (len(subsets), len(valid_operators))) if debug>1 else None
        # Access the next subset
        H = np.where(subsets[0,:-1])[0]
        cond_1 = subsets[0,-1]
        subsets = subsets[1:]
        # Check if the validity condition holds for H, i.e. that
        # NA_yx \ H is a clique.
        # If it has not been tested previously for a subset of H,
        # check it now
        if not cond_1 and utils.is_clique(na_yx - set(H), A):
            cond_1 = True
            # For all supersets H' of H, the validity condition will also hold
            supersets = subsets[:,H].all(axis=1)
            subsets[supersets,-1] = True
        # If the validity condition holds, apply operator and compute its score
        print("      delete(%d,%d,%s)"%(x,y,H), "na_yx - H = ", na_yx - set(H), "validity:", cond_1) if debug>1 else None
        if cond_1:
            # Apply operator
            new_A = delete(x,y,H,A)
            # Compute the change in score
            aux = (na_yx - set(H)) | utils.pa(y, A) | {x}
            #print(x,y,H,"na_yx:",na_yx,"old:",aux,"new:", aux - {x})
            old_score = cache.local_score(y, aux)
            new_score = cache.local_score(y, aux - {x})
            print("        new: s(%d, %s) = %0.6f old: s(%d, %s) = %0.6f" % (y, aux - {x}, new_score, y, aux, old_score)) if debug>1 else None
            # Add to the list of valid operators
            valid_operators.append((new_score - old_score, new_A, x, y, H))
            print("    delete(%d,%d,%s) -> %0.16f" % (x,y,H,new_score - old_score)) if debug else None
    # Return all the valid operators
    return valid_operators

# --------------------------------------------------------------------
# Insert operator
#    1. definition in function insert
#    2. enumeration logic (to enumerate and score only valid
#    operators) function in score_valid_insert_operators

def insert(x,y,T,A):
    """
    Applies the insert operator:
      1) adds the edge x -> y
      2) for all t in T, orients the previously undirected edge t -> y

    Parameters
    ----------
    x : int
        the origin node (i.e. x -> y)
    y : int
        the target node
    T : iterable of ints
        a subset of the neighbors of y which are not adjacent to x
    A : np.array
        the current adjacency matrix
    
    Returns
    -------
    new_A : np.array
        the adjacency matrix resulting from applying the operator

    """
    # Check inputs
    T = sorted(T)
    if A[x,y] !=0 or A[y,x] !=0:
        raise ValueError("x=%d and y=%d are already connected" % (x,y))
    if len(T) == 0:
        pass
    elif not (A[T,y].all() and A[y,T].all()):
        raise ValueError("Not all nodes in T=%s are neighbors of y=%d" % (T,y))
    elif A[T,x].any() or A[x,T].any():
        raise ValueError("Some nodes in T=%s are adjacent to x=%d" % (T,x))
    # Apply operator
    new_A = A.copy()
    # Add edge x -> y
    new_A[x,y] = 1
    # Orient edges t - y to t -> y, for t in T
    new_A[T, y] = 1
    new_A[y, T] = 0
    return new_A

def score_valid_insert_operators(x,y,A,cache,debug=0):
    """Generate and score all valid insert(x,y,T) operators involving the edge
    x-> y, and all possible subsets T of neighbors of y which
    are NOT adjacent to x.
   
    Parameters
    ----------
    x : int
        the origin node (i.e. x -> y)
    y : int
        the target node
    A : np.array
        the current adjacency matrix
    cache : instance of ges.scores.DecomposableScore
        the score cache to compute the score of the
        operators that are valid
    debug : bool or string
        if debug traces should be printed (True/False). If a non-empty
        string is passed, traces are printed with the given string as
        prefix (useful for indenting the prints from a calling
        function)

    Returns
    -------
    valid_operators : list of tuples
        a list of tubles, each containing a valid operator, its score
        and the resulting connectivity matrix

    """
    p = len(A)
    if A[x,y] != 0 or A[y,x] != 0:
        raise ValueError("x=%d and y=%d are already connected" % (x,y))
    # One-hot encode all subsets of T0, plus one extra column to mark
    # if they pass validity condition 2 (see below)
    T0 = sorted(utils.neighbors(y,A) - utils.adj(x,A))
    if len(T0) == 0:
        subsets = np.zeros((1,p+1), dtype=np.bool)
    else:
        subsets = np.zeros((2**len(T0), p+1), dtype=np.bool)
        subsets[:, T0] = utils.cartesian([np.array([False, True])] * len(T0), dtype=np.bool)
    valid_operators = []
    print("    insert(%d,%d) T0=" % (x,y),set(T0)) if debug>1 else None
    while len(subsets) > 0:
        print("      len(subsets)=%d, len(valid_operators)=%d" % (len(subsets), len(valid_operators))) if debug>1 else None
        # Access the next subset
        T = np.where(subsets[0,:-1])[0]
        passed_cond_2 = subsets[0,-1]
        subsets = subsets[1:]
        # Check that the validity conditions hold for T
        na_yxT = utils.na(y,x,A) | set(T)
        # Condition 1: Test that NA_yx U T is a clique
        cond_1 = utils.is_clique(na_yxT, A)
        if not cond_1:
            # Remove from consideration all other sets T' which
            # contain T, as the clique condition will also not hold
            supersets = subsets[:,T].all(axis=1)
            subsets = np.delete(subsets, supersets, axis=0)
        # Condition 2: Test that all semi-directed paths from y to x contain a
        # member from NA_yx U T
        if passed_cond_2:
            # If a subset of T satisfied condition 2, so does T
            cond_2 = True
        else:
            # Check condition 2
            cond_2 = True
            for path in utils.semi_directed_paths(y,x,A):
                if len(na_yxT & set(path)) == 0:
                    cond_2 = False
                    break
            if cond_2:
                # If condition 2 holds for NA_yx U T, then it holds for all supersets of T
                supersets = subsets[:,T].all(axis=1)
                subsets[supersets,-1] = True
        print("      insert(%d,%d,%s)"%(x,y,T), "na_yx U T = ", na_yxT, "validity:", cond_1, cond_2) if debug>1 else None
        # If both conditions hold, apply operator and compute its score
        if cond_1 and cond_2:
            # Apply operator
            new_A = insert(x,y,T,A)
            # Compute the change in score
            aux = na_yxT | utils.pa(y, A)
            old_score = cache.local_score(y, aux)
            new_score = cache.local_score(y, aux | {x})
            print("        new: s(%d, %s) = %0.6f old: s(%d, %s) = %0.6f" % (y, aux | {x}, new_score, y, aux, old_score)) if debug>1 else None
            # Add to the list of valid operators
            valid_operators.append((new_score - old_score, new_A, x, y, T))
            print("    insert(%d,%d,%s) -> %0.16f" % (x,y,T,new_score - old_score)) if debug else None
    # Return all the valid operators
    return valid_operators

# --------------------------------------------------------------------
# Turn operator
#    1. definition in function turn
#    2. enumeration logic (to enumerate and score only valid
#    operators) function in score_valid_insert_operators

def turn(x,y,C,A):
    """
    Applies the turning operator: For an edge x - y or x <- y,
      1) orients the edge as x -> y
      2) for all c in C, orients the previously undirected edge c -> y

    Parameters
    ----------
    x : int
        the origin node (i.e. x -> y)
    y : int
        the target node
    C : iterable of ints
        a subset of the neighbors of y
    A : np.array
        the current adjacency matrix
    
    Returns
    -------
    new_A : np.array
        the adjacency matrix resulting from applying the operator

    """
    # Check inputs
    if A[x,y] !=0 and A[y,x] == 0:
        raise ValueError("The edge %d -> %d is already exists" % (x,y))
    if A[x,y] == 0 and A[y,x] == 0:
        raise ValueError("x=%d and y=%d are not connected" % (x,y))
    if not C <= utils.neighbors(y,A):
        raise ValueError("Not all nodes in C=%s are neighbors of y=%d" % (C,y))
    if len({x,y} & C) > 0:
        raise ValueError("C should not contain x or y")
    # Apply operator
    new_A = A.copy()
    # Turn edge x -> y
    new_A[y,x] = 0
    new_A[x,y] = 1
    # Orient edges c -> y for c in C
    new_A[y, list(C)] = 0
    return new_A

def score_valid_turn_operators(x,y,A,cache,debug=0):
    """Generate and score all valid turn(x,y,C) operators involving the edge
    x -> y, and all possible subsets C of neighbors of y.
   
    Parameters
    ----------
    x : int
        the origin node (i.e. x -> y)
    y : int
        the target node
    A : np.array
        the current adjacency matrix
    cache : instance of ges.scores.DecomposableScore
        the score cache to compute the score of the
        operators that are valid
    debug : bool or string
        if debug traces should be printed (True/False). If a non-empty
        string is passed, traces are printed with the given string as
        prefix (useful for indenting the prints from a calling
        function)

    Returns
    -------
    valid_operators : list of tuples
        a list of tubles, each containing a valid operator, its score
        and the resulting connectivity matrix

    """
    p = len(A)
    if A[x,y] != 0 and A[y,x] == 0:
        raise ValueError("The edge %d -> %d already exists" % (x,y))
    if A[x,y] == 0 and A[y,x] == 0:
        raise ValueError("x=%d and y=%d are not connected" % (x,y))
    # Different validation/scoring logic when the edge to be turned is
    # essential (x <- x) or not (x - y)
    if A[x,y] != 0 and A[y,x] != 0:
        return score_valid_turn_operators_undir(x,y,A,cache,debug=debug)
    else:
        return score_valid_turn_operators_dir(x,y,A,cache,debug=debug)

def score_valid_turn_operators_dir(x,y,A,cache,debug=0):
    """Logic for finding and scoring the valid turn operators that can be
    applied to the edge x <- y.

    Parameters
    ----------
    x : int
        the origin node (i.e. x -> y)
    y : int
        the target node
    A : np.array
        the current adjacency matrix
    cache : instance of ges.scores.DecomposableScore
        the score cache to compute the score of the
        operators that are valid
    debug : bool or string
        if debug traces should be printed (True/False). If a non-empty
        string is passed, traces are printed with the given string as
        prefix (useful for indenting the prints from a calling
        function)

    Returns
    -------
    valid_operators : list of tuples
        a list of tubles, each containing a valid operator, its score
        and the resulting connectivity matrix

    """
    # One-hot encode all subsets of T0, plus one extra column to mark
    # if they pass validity condition 2 (see below). The set C passed
    # to the turn operator will be C = NAyx U T.
    p = len(A)
    T0 = sorted(utils.neighbors(y,A) - utils.adj(x,A))
    if len(T0) == 0:
        subsets = np.zeros((1,p+1), dtype=np.bool)
    else:
        subsets = np.zeros((2**len(T0), p+1), dtype=np.bool)
        subsets[:, T0] = utils.cartesian([np.array([False, True])] * len(T0), dtype=np.bool)
    valid_operators = []
    print("    turn(%d,%d) T0=" % (x,y),set(T0)) if debug>1 else None
    while len(subsets) > 0:
        print("      len(subsets)=%d, len(valid_operators)=%d" % (len(subsets), len(valid_operators))) if debug>1 else None
        # Access the next subset
        T = np.where(subsets[0,:-1])[0]
        passed_cond_2 = subsets[0,-1]
        subsets = subsets[1:] # update the list of remaining subsets
        # Check that the validity conditions hold for T
        C = utils.na(y,x,A) | set(T)
        # Condition 1: Test that C = NA_yx U T is a clique
        cond_1 = utils.is_clique(C, A)
        if not cond_1:
            # Remove from consideration all other sets T' which
            # contain T, as the clique condition will also not hold
            supersets = subsets[:,T].all(axis=1)
            subsets = np.delete(subsets, supersets, axis=0)
        # Condition 2: Test that all semi-directed paths from y to x contain a
        # member from C U neighbors(x)
        if passed_cond_2:
            # If a subset of T satisfied condition 2, so does T
            cond_2 = True
        else:
            # otherwise, check condition 2
            cond_2 = True
            for path in utils.semi_directed_paths(y,x,A):
                if path == [y,x]:
                    pass
                elif len((C | utils.neighbors(x, A)) & set(path)) == 0:
                    cond_2 = False
                    break
            if cond_2:
                # If condition 2 holds for C U neighbors(x), that is,
                # for C = NAyx U T U neighbors(x), then it holds for
                # all supersets of T
                supersets = subsets[:,T].all(axis=1)
                subsets[supersets,-1] = True
        # If both conditions hold, apply operator and compute its score
        print("      turn(%d,%d,%s)"%(x,y,C), "na_yx =", utils.na(y,x,A), "T =",T, "validity:", cond_1, cond_2) if debug>1 else None
        if cond_1 and cond_2:
            # Apply operator
            new_A = turn(x,y,C,A)
            # Compute the change in score
            new_score = cache.local_score(y, utils.pa(y, A) | C | {x}) + cache.local_score(x, utils.pa(x,A) - {y})
            old_score = cache.local_score(y, utils.pa(y, A) | C) + cache.local_score(x, utils.pa(x,A))
            print("        new score = %0.6f, old score = %0.6f, y=%d, C=%s" % (new_score, old_score, y, C)) if debug>1 else None
            # Add to the list of valid operators
            valid_operators.append((new_score - old_score, new_A, x, y, C))
            print("    turn(%d,%d,%s) -> %0.16f" % (x,y,C,new_score - old_score)) if debug else None
    # Return all the valid operators
    return valid_operators

def score_valid_turn_operators_undir(x,y,A,cache,debug=0):
    """Logic for finding and scoring the valid turn operators that can be
    applied to the edge x - y.

    Parameters
    ----------
    x : int
        the origin node (i.e. x -> y)
    y : int
        the target node
    A : np.array
        the current adjacency matrix
    cache : instance of ges.scores.DecomposableScore
        the score cache to compute the score of the
        operators that are valid
    debug : bool or string
        if debug traces should be printed (True/False). If a non-empty
        string is passed, traces are printed with the given string as
        prefix (useful for indenting the prints from a calling
        function)

    Returns
    -------
    valid_operators : list of tuples
        a list of tubles, each containing a valid operator, its score
        and the resulting connectivity matrix

    """
    # Proposition 31, condition (ii) in GIES paper (Hauser & Bühlmann
    # 2012) is violated if:
    #   1. all neighbors of y are adjacent to x, or
    #   2. y has no neighbors (besides u)
    # then there are no valid operators.
    non_adjacents = list(utils.neighbors(y,A) - utils.adj(x,A) - {x})
    if len(non_adjacents) == 0:
        print("    turn(%d,%d) : ne(y) \ adj(x) = Ø => stopping" % (x,y)) if debug>1 else None
        return []
    # Otherwise, construct all the possible subsets which will satisfy
    # condition (ii), i.e. all subsets of neighbors of y with at least
    # one which is not adjacent to x
    p = len(A)
    C0 = sorted(utils.neighbors(y,A) - {x})
    subsets = np.zeros((2**len(C0), p+1), dtype=np.bool)
    subsets[:, C0] = utils.cartesian([np.array([False,True])] * len(C0), dtype=np.bool)
    # Remove all subsets which do not contain at least one non-adjacent node to x
    to_remove = (subsets[:,non_adjacents] == False).all(axis=1)
    subsets = np.delete(subsets, to_remove, axis=0)
    # With condition (ii) guaranteed, we now check conditions (i,iii)
    # for each subset
    valid_operators = []
    print("    turn(%d,%d) C0=" % (x,y),set(C0)) if debug>1 else None
    while len(subsets) > 0:
        print("      len(subsets)=%d, len(valid_operators)=%d" % (len(subsets), len(valid_operators))) if debug>1 else None
        # Access the next subset
        C = set(np.where(subsets[0,:])[0])
        subsets = subsets[1:]
        # Condition (i): C is a clique in the subgraph induced by the
        # chain component of y. Because C is composed of neighbors of
        # y, this is equivalent to C being a clique in A. NOTE: This
        # is also how it is described in Alg. 5 of the paper
        cond_1 = utils.is_clique(C, A)
        if not cond_1:
            # Remove from consideration all other sets C' which
            # contain C, as the clique condition will also not hold
            supersets = subsets[:,list(C)].all(axis=1)
            subsets = np.delete(subsets, supersets, axis=0)
            continue
        # Condition (iii): (NOTE: THIS APPEARS TO BE WRONG, USING
        # CONDITION iii FROM Alg. 5 INSTEAD) Let na_yx (N in the GIES
        # paper) be the neighbors of Y which are adjacent to X. Then,
        # the intersection S between C and na_yx (i.e. the members of
        # C which are adjacent to X), separates C \ na_yx from na_yx \
        # C, i.e. all the paths from one set to the other contain a
        # node in S.
        subgraph = utils.induced_subgraph(utils.chain_component(y, A), A)
        na_yx = utils.na(y, x, A)
        if not utils.separates({x,y}, C, na_yx - C, subgraph):
            continue
        # At this point C passes both conditions
        #   Apply operator
        new_A = turn(x,y,C,A)
        #   Compute the change in score
        new_score = cache.local_score(y, utils.pa(y, A) | C | {x}) + cache.local_score(x, utils.pa(x,A) | (C & na_yx))
        old_score = cache.local_score(y, utils.pa(y, A) | C) + cache.local_score(x, utils.pa(x,A) | (C & na_yx) | {y})
        print("        new score = %0.6f, old score = %0.6f, y=%d, C=%s" % (new_score, old_score, y, C)) if debug>1 else None
        #   Add to the list of valid operators
        valid_operators.append((new_score - old_score, new_A, x, y, C))
        print("    turn(%d,%d,%s) -> %0.16f" % (x,y,C,new_score - old_score)) if debug else None
    # Return all valid operators
    return valid_operators

