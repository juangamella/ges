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

import unittest
import numpy as np
import sempler
import research.utils as utils
import causaldag as cd # to obtain cpdag from dag

import ges.utils

#---------------------------------------------------------------------
class PDAG_to_CPDAG_Tests(unittest.TestCase):
    # Tests to ensure that the conversion from PDAG to CPDAG
    # works
    
    def test_pdag_to_dag_1(self):
        # Should work
        P = np.array([[0,0,1,0],
                      [0,0,1,1],
                      [1,0,0,0],
                      [0,0,0,0]])
        A = ges.utils.pdag_to_dag(P, debug=False)
        #print(A)
        true_A = P.copy()
        true_A[0,2] = 0
        self.assertTrue((A == true_A).all())

    def test_pdag_to_dag_2(self):
        # Same as above but different index order, should work
        P = np.array([[0,0,1,0],
                      [1,0,0,1],
                      [1,0,0,0],
                      [0,0,0,0]])
        A = ges.utils.pdag_to_dag(P, debug=False)
        #print(A)
        true_A = P.copy()
        true_A[2,0] = 0
        self.assertTrue((A == true_A).all())
        
    def test_pdag_to_dag_3(self):
        # Should work
        P = np.array([[0,0,1,0],
                      [0,0,0,1],
                      [1,1,0,0],
                      [0,0,0,0]])
        A = ges.utils.pdag_to_dag(P, debug=False)
        #print(A)
        true_A1, true_A2 = P.copy(), P.copy()
        true_A1[0,2], true_A2[2,0] = 0,0
        self.assertTrue(utils.member([true_A1, true_A2], A) is not None)

    def test_pdag_to_dag_4(self):
        # This PDAG does not admit a consistent extension, i.e. it
        # either creates a non-existing v-structure or it induces a
        # cycle
        P = np.array([[0,0,1,1],
                      [0,0,1,0],
                      [1,0,0,0],
                      [0,0,1,0]])
        try:
            ges.utils.pdag_to_dag(P, debug=False)
            self.fail("Exception should have been thrown")
        except ValueError as e:
            print("OK:", e)

    def test_pdag_to_dag_5(self):
        # Fully directed PDAG should return itself
        A = np.array([[0,1,1,0],
                      [0,0,1,0],
                      [0,0,0,0],
                      [0,0,1,0]])
        extension = ges.utils.pdag_to_dag(A, debug=False)
        self.assertTrue(utils.is_consistent_extension(extension, A))
        self.assertTrue((extension == A).all())
        
    def test_pdag_to_dag_6(self):
        # Check that the resulting extensions are indeed a consistent
        # extensions
        G = 500
        p = 20
        for i in range(G):
            A = sempler.dag_avg_deg(p,3,1,1)
            cpdag = cd.DAG.from_amat(A).cpdag().to_amat()[0]
            self.assertTrue(utils.is_consistent_extension(A, cpdag))
            extension = ges.utils.pdag_to_dag(cpdag, debug=False)
            is_consistent_extension = utils.is_consistent_extension(extension, cpdag)
            if not is_consistent_extension:
                print("DAG\n", A)
                print("CPDAG\n", cpdag)
                print("Extension\n",extension)
                utils.is_consistent_extension(extension, cpdag, debug=True)
                # Rerun with outputs
                assert (extension == ges.utils.pdag_to_dag(cpdag, debug=True)).all()
                self.assertTrue(is_consistent_extension)
        print("\nChecked PDAG to DAG conversion for %d PDAGs" % (i+1))

    def test_order_edges_preconditions(self):
        A = np.array([[0, 0, 1, 1, 1]
                      ,[0, 0, 1, 1, 1]
                      ,[0, 0, 0, 1, 1]
                      ,[0, 0, 0, 0, 1]
                      ,[0, 0, 0, 0, 0]])
        # Check that exception is thrown with pdag
        pdag = A.copy()
        pdag[4,2] = 1
        try:
            ges.utils.order_edges(pdag)
            self.fail()
        except ValueError as e:
            print("OK:",e)
        # Check that exception is thrown with cyclic graph
        cyclic = pdag.copy()
        cyclic[2,4] = 0
        try:
            ges.utils.order_edges(cyclic)
            self.fail()
        except ValueError as e:
            print("OK:",e)
        

    def test_order_edges_1(self):
        A = np.array([[0, 0, 1, 1, 1]
                      ,[0, 0, 1, 1, 1]
                      ,[0, 0, 0, 1, 1]
                      ,[0, 0, 0, 0, 1]
                      ,[0, 0, 0, 0, 0]])
        ordered = ges.utils.order_edges(A)
        #print(ordered)
        # Ground truth derived by hand for the order [1,0,2,3,4]
        truth = np.array([[0, 0, 9, 6, 2],
                          [0, 0, 8, 5, 1],
                          [0, 0, 0, 7, 3],
                          [0, 0, 0, 0, 4],
                          [0, 0, 0, 0, 0]])
        self.assertTrue((ordered >= 0).all())
        self.assertTrue((ordered == truth).all())

    def test_order_edges_2(self):
        G = 500
        p = 20
        for i in range(G):
            A = sempler.dag_avg_deg(p,3,1,1)
            ordered = ges.utils.order_edges(A)
            no_edges = (A != 0).sum()
            self.assertEqual(sorted(ordered[ordered != 0]), list(range(1,no_edges+1)))
        print("\nChecked valid ordering for %d DAGs" % (i+1))


    def test_label_edges_preconditions(self):
        A = np.array([[0, 0, 1, 1, 1]
                      ,[0, 0, 1, 1, 1]
                      ,[0, 0, 0, 1, 1]
                      ,[0, 0, 0, 0, 1]
                      ,[0, 0, 0, 0, 0]])
        # Check that exception is thrown with pdag
        pdag = A.copy()
        pdag[4,2] = 1
        try:
            ges.utils.order_edges(pdag)
            self.fail()
        except ValueError as e:
            print("OK:",e)
        # Check that exception is thrown with cyclic graph
        cyclic = pdag.copy()
        cyclic[2,4] = 0
        try:
            ges.utils.order_edges(cyclic)
            self.fail()
        except ValueError as e:
            print("OK:",e)
        # Check that if ordering is invalid an exception is thrown
        try:
            ges.utils.label_edges(A)
            self.fail()
        except ValueError as e:
            print("OK:",e)
        # Same same, but different :)
        ordered = ges.utils.order_edges(A)
        ordered[0,4] = 1
        try:
            ges.utils.label_edges(ordered)
            self.fail()
        except ValueError as e:
            print("OK:",e)
        
    def test_label_edges_1(self):
        # For a hand-picked example
        A = np.array([[0,0,1,0,0],
                      [0,0,1,0,0],
                      [0,0,0,1,1],
                      [0,0,0,0,1],
                      [0,0,0,0,0]])
        ordered = ges.utils.order_edges(A)
        truth = np.array([[0,0,1,0,0],
                          [0,0,1,0,0],
                          [0,0,0,1,1],
                          [0,0,0,0,-1],
                          [0,0,0,0,0]])
        labelled = ges.utils.label_edges(ordered)
        self.assertTrue((labelled == truth).all())

    def test_label_edges_2(self):
        # With randomly generated DAGs
        #np.random.seed(42)
        G = 500
        p = 20
        for i in range(G):
            A = sempler.dag_avg_deg(p,3,1,1)
            # Construct expected output
            cpdag = cd.DAG.from_amat(A).cpdag().to_amat()[0]
            undirected = np.logical_and(cpdag, cpdag.T)
            truth = A.copy()
            truth[np.logical_and(truth, undirected)] = -1
            # Run and assert
            ordered = ges.utils.order_edges(A)
            labelled = ges.utils.label_edges(ordered)
            self.assertTrue((labelled == truth).all())
        print("\nChecked edge labelling for %d DAGs" % (i+1))

    def test_dag_to_cpdag(self):
        # Test by checking that applying the whole pipeline to a CPDAG
        #returns the same CPDAG
        G = 500
        p = 25
        for i in range(G):
            A = sempler.dag_avg_deg(p,4,1,1)
            truth = cd.DAG.from_amat(A).cpdag().to_amat()[0]
            # Run and assert
            cpdag = ges.utils.dag_to_cpdag(A)
            self.assertTrue((truth == cpdag).all())
        print("\nChecked DAG to CPDAG conversion for %d DAGs" % (i+1))

    def test_cpdag_to_cpdag(self):
        # Test by checking that applying the whole pipeline to a CPDAG
        # returns the same CPDAG
        G = 500
        p = 30
        for i in range(G):
            A = sempler.dag_avg_deg(p,3,1,1)
            cpdag = cd.DAG.from_amat(A).cpdag().to_amat()[0]
            # Run and assert
            output = ges.utils.pdag_to_cpdag(cpdag)
            self.assertTrue((output == cpdag).all())
        print("\nChecked CPDAG to CPDAG conversion for %d CPDAGs" % (i+1))

    def test_pdag_to_cpdag(self):
        # Now construct PDAGs whose extensions belong to the true MEC,
        # and test that the true CPDAG is recovered
        G = 500
        p = 32
        for g in range(G):
            A = sempler.dag_avg_deg(p,3,1,1)
            # Construct PDAG by undirecting random edges which do not
            # belong to a v-structure.
            # NOTE: I'm proceeding in this awkward way to avoid
            # using functions from the pipeline I'm testing,
            # i.e. ges.utils.order_edges and ges.utils.label_edges
            pdag = A.copy()
            mask_vstructs = np.zeros_like(A)
            for (i,c,j) in utils.vstructures(A):
                mask_vstructs[i,c] = 1
                mask_vstructs[j,c] = 1
            flippable = np.logical_and(A, np.logical_not(mask_vstructs))
            fros,tos = np.where(flippable)
            for (x,y) in zip(fros,tos):
                # Undirect approximately 2/3 of the possible edges
                if np.random.binomial(1, 2/3):
                    pdag[y,x] = 1
            # Run and assert
            self.assertTrue(utils.is_consistent_extension(A,pdag))
            truth = cd.DAG.from_amat(A).cpdag().to_amat()[0]
            output = ges.utils.pdag_to_cpdag(pdag)
            self.assertTrue((output == truth).all())
        print("\nChecked PDAG to CPDAG conversion for %d PDAGs" % (g+1))

    def test_maximally_orient_1(self):
        graphs = [
            # Graph 1 (Meek rule 1)
            (np.array([[0,1,0],
                       [0,0,1],
                       [0,1,0]]),
             np.array([[0,1,0],
                       [0,0,1],
                       [0,0,0]])),
            # Graph 2 (Meek rule 1)
            (np.array([[0,1,1,0],
                       [0,0,1,0],
                       [0,1,0,0],
                       [0,0,1,0]]),
             np.array([[0,1,1,0],
                       [0,0,0,0],
                       [0,1,0,0],
                       [0,0,1,0]])),
            # Graph 3 (Meek rule 1)
            (np.array([[0,1,1],
                       [0,0,1],
                       [0,1,0]]),
             np.array([[0,1,1],
                       [0,0,1],
                       [0,1,0]])),
            # Graph 4 (Meek rule 1)
            (np.array([[0,0,1,0,0],
                       [0,0,1,0,0],
                       [0,0,0,1,1],
                       [0,0,1,0,0],
                       [0,0,1,0,0]]),
             np.array([[0,0,1,0,0],
                       [0,0,1,0,0],
                       [0,0,0,1,1],
                       [0,0,0,0,0],
                       [0,0,0,0,0]])),
            # Graph 5 (Meek rule 2)
            (np.array([[0,1,1],
                       [0,0,1],
                       [1,0,0]]),
             np.array([[0,1,1],
                       [0,0,1],
                       [0,0,0]])),
            # Graph 6 (Meek rule 2)
            (np.array([[0,1,1,1],
                       [0,0,0,1],
                       [0,0,0,1],
                       [1,0,0,0]]),
             np.array([[0,1,1,1],
                       [0,0,0,1],
                       [0,0,0,1],
                       [0,0,0,0]])),
            # Graph 7 (Meek rule 2)
            (np.array([[0,1,1],
                       [0,0,0],
                       [1,1,0]]),
             np.array([[0,1,1],
                       [0,0,0],
                       [1,1,0]])),
            # Graph 8 (No consistent extension)
            (np.array([[0,1,0,1,0],
                       [0,0,1,1,0],
                       [0,0,0,0,1],
                       [1,0,0,0,1],
                       [0,0,1,1,0]]),
              None),
            # Graph 9 (Meek rule 3)
            (np.array([[0,1,1,1],
                       [1,0,0,1],
                       [1,0,0,1],
                       [1,0,0,0]]),
             np.array([[0,1,1,1],
                       [1,0,0,1],
                       [1,0,0,1],
                       [0,0,0,0]])),
            # Graph 10
            (np.array([[0,1,1,1],
                       [1,0,0,1],
                       [0,0,0,1],
                       [1,0,0,0]]),
             np.array([[0,1,1,1],
                       [1,0,0,1],
                       [0,0,0,1],
                       [0,0,0,0]])),
            # Graph 11 (No consistent extension)
            (np.array([[0,1,1,1,1,0],
                       [1,0,0,1,1,0],
                       [1,0,0,1,1,0],
                       [1,0,0,0,0,0],
                       [1,0,0,0,0,0],
                       [0,0,0,0,1,0]]),
             None),
            # Graph 12 (Meek rule 4)
            (np.array([[0,1,1,0],
                       [1,0,1,1],
                       [0,1,0,1],
                       [0,1,0,0]]),
             np.array([[0,1,1,0],
                       [1,0,1,1],
                       [0,1,0,1],
                       [0,0,0,0]])),
            # Graph 13 (Meek rule 4)
            (np.array([[0,1,0,0,1],
                       [0,0,1,0,1],
                       [0,0,0,1,1],
                       [0,0,0,0,1],
                       [1,1,1,1,0]]),
             np.array([[0,1,0,0,1],
                       [0,0,1,0,1],
                       [0,0,0,1,0],
                       [0,0,0,0,0],
                       [1,1,1,1,0]]))

        ]
        # Test
        for (A, truth) in graphs:
            if truth is None:
                try:
                    ges.utils.maximally_orient(A)
                    self.fail("Exception should have been thrown")
                except Exception as e:
                    print("OK:", e)
            else:
                self.assertTrue((truth == ges.utils.maximally_orient(A)).all())

    def test_maximally_orient_2(self):
        G = 500
        p = 30
        for k in range(G):
            A = sempler.dag_avg_deg(p,3,1,1)
            pdag = ges.utils.pdag_to_cpdag(A)
            fro,to = np.where(utils.only_undirected(pdag) != 0)
            if len(fro) == 0:
                continue
            orient = np.random.choice(len(fro), np.random.randint(0,len(fro)))
            for idx in orient:
                i,j = fro[idx], to[idx]
                if np.random.binomial(1,0.5):
                    pdag[i,j] = 0
                else:
                    pdag[i,j] = 0
            # Result from succesively applying meek rules should have
            # the same skeleton, v-structures and respect existing
            # edge orientations
            try:
                oriented = ges.utils.maximally_orient(pdag)
                self.assertTrue(utils.is_consistent_extension(oriented, pdag))
            except ValueError:
                pass
        print("\nChecked meek rule orientation for %d PDAGs" % (k+1))

                
    def test_meek_rule_1_a(self):
        A = np.array([[0,1,0],
                      [0,0,1],
                      [0,1,0]])
        self.assertTrue(ges.utils.rule_1(1,2,A))
        self.assertFalse(ges.utils.rule_1(2,1,A))
        # Other rules should not orient the edge
        self.assertFalse(ges.utils.rule_2(1,2,A))
        self.assertFalse(ges.utils.rule_2(2,1,A))
        self.assertFalse(ges.utils.rule_3(1,2,A))
        self.assertFalse(ges.utils.rule_3(2,1,A))
        self.assertFalse(ges.utils.rule_4(1,2,A))
        self.assertFalse(ges.utils.rule_4(2,1,A))
                    
    def test_meek_rule_1_b(self):
        A = np.array([[0,1,1,0],
                      [0,0,1,0],
                      [0,1,0,0],
                      [0,0,1,0]])
        self.assertTrue(ges.utils.rule_1(2,1,A))
        self.assertFalse(ges.utils.rule_1(1,2,A))
        # Other rules should not orient the edge
        self.assertFalse(ges.utils.rule_2(1,2,A))
        self.assertFalse(ges.utils.rule_2(2,1,A))
        self.assertFalse(ges.utils.rule_3(1,2,A))
        self.assertFalse(ges.utils.rule_3(2,1,A))
        self.assertFalse(ges.utils.rule_4(1,2,A))
        self.assertFalse(ges.utils.rule_4(2,1,A))

    def test_meek_rule_1_c(self):
        A = np.array([[0,1,1],
                       [0,0,1],
                       [0,1,0]])
        self.assertFalse(ges.utils.rule_1(1,2,A))
        self.assertFalse(ges.utils.rule_1(2,1,A))
        # Other rules should not orient the edge
        self.assertFalse(ges.utils.rule_2(1,2,A))
        self.assertFalse(ges.utils.rule_2(2,1,A))
        self.assertFalse(ges.utils.rule_3(1,2,A))
        self.assertFalse(ges.utils.rule_3(2,1,A))
        self.assertFalse(ges.utils.rule_4(1,2,A))
        self.assertFalse(ges.utils.rule_4(2,1,A))

    def test_meek_rule_1_d(self):
        A = np.array([[0,0,1,0,0],
                      [0,0,1,0,0],
                      [0,0,0,1,1],
                      [0,0,1,0,0],
                      [0,0,1,0,0]])
        self.assertTrue(ges.utils.rule_1(2,3,A))
        self.assertFalse(ges.utils.rule_1(3,2,A))
        self.assertTrue(ges.utils.rule_1(2,4,A))
        self.assertFalse(ges.utils.rule_1(4,2,A))
        # Other rules should not orient the edges
        self.assertFalse(ges.utils.rule_2(2,3,A))
        self.assertFalse(ges.utils.rule_2(3,2,A))
        self.assertFalse(ges.utils.rule_2(2,4,A))
        self.assertFalse(ges.utils.rule_2(4,2,A))
        self.assertFalse(ges.utils.rule_3(2,3,A))
        self.assertFalse(ges.utils.rule_3(3,2,A))
        self.assertFalse(ges.utils.rule_3(2,4,A))
        self.assertFalse(ges.utils.rule_3(4,2,A))
        self.assertFalse(ges.utils.rule_4(2,3,A))
        self.assertFalse(ges.utils.rule_4(3,2,A))
        self.assertFalse(ges.utils.rule_4(2,4,A))
        self.assertFalse(ges.utils.rule_4(4,2,A))
        
    def test_meek_rule_2_a(self):
        A = np.array([[0,1,1],
                      [0,0,1],
                      [1,0,0]])
        self.assertTrue(ges.utils.rule_2(0,2,A))
        self.assertFalse(ges.utils.rule_2(2,0,A))
        # Other rules should not orient the edge
        self.assertFalse(ges.utils.rule_1(0,2,A))
        self.assertFalse(ges.utils.rule_1(2,0,A))
        self.assertFalse(ges.utils.rule_3(0,2,A))
        self.assertFalse(ges.utils.rule_3(2,0,A))
        self.assertFalse(ges.utils.rule_4(0,2,A))
        self.assertFalse(ges.utils.rule_4(2,0,A))

    def test_meek_rule_2_b(self):
        A = np.array([[0,1,1,1],
                      [0,0,0,1],
                      [0,0,0,1],
                      [1,0,0,0]])
        self.assertTrue(ges.utils.rule_2(0,3,A))
        self.assertFalse(ges.utils.rule_2(3,0,A))
        # Other rules should not orient the edge
        self.assertFalse(ges.utils.rule_1(0,3,A))
        self.assertFalse(ges.utils.rule_1(3,0,A))
        self.assertFalse(ges.utils.rule_3(0,3,A))
        self.assertFalse(ges.utils.rule_3(3,0,A))
        self.assertFalse(ges.utils.rule_4(0,3,A))
        self.assertFalse(ges.utils.rule_4(3,0,A))
        
    def test_meek_rule_2_c(self):
        A = np.array([[0,1,1],
                      [0,0,0],
                      [1,1,0]])
        self.assertFalse(ges.utils.rule_2(0,2,A))
        self.assertFalse(ges.utils.rule_2(2,0,A))
        # Other rules should not orient the edge
        self.assertFalse(ges.utils.rule_1(0,2,A))
        self.assertFalse(ges.utils.rule_1(2,0,A))
        self.assertFalse(ges.utils.rule_3(0,2,A))
        self.assertFalse(ges.utils.rule_3(2,0,A))
        self.assertFalse(ges.utils.rule_4(0,2,A))
        self.assertFalse(ges.utils.rule_4(2,0,A))

    def test_meek_rule_2_d(self):
        A = np.array([[0,1,0,1,0],
                      [0,0,1,1,0],
                      [0,0,0,0,1],
                      [1,0,0,0,1],
                      [0,0,1,1,0]])
        # edge 0 - 3 should orient to 0 -> 3
        self.assertTrue(ges.utils.rule_2(0,3,A))
        self.assertFalse(ges.utils.rule_2(3,0,A))
        # edge 2 - 4 should not be oriented
        self.assertFalse(ges.utils.rule_2(2,4,A))
        self.assertFalse(ges.utils.rule_2(4,2,A))
        # edge 3 - 4 should not be oriented
        self.assertFalse(ges.utils.rule_2(3,4,A))
        self.assertFalse(ges.utils.rule_2(4,3,A))

    def test_meek_rule_3_a(self):
        A = np.array([[0,1,1,1],
                      [1,0,0,1],
                      [1,0,0,1],
                      [1,0,0,0]])
        self.assertTrue(ges.utils.rule_3(0,3,A))
        self.assertFalse(ges.utils.rule_3(3,0,A))
        # The other edges should not be oriented by rule 3
        self.assertFalse(ges.utils.rule_3(0,1,A))
        self.assertFalse(ges.utils.rule_3(1,0,A))
        self.assertFalse(ges.utils.rule_3(0,2,A))
        self.assertFalse(ges.utils.rule_3(2,0,A))
        # None of the edges should be oriented by the other rules
        for i,j in [(0,3),(0,1),(0,2)]:
            self.assertFalse(ges.utils.rule_1(i,j,A))
            self.assertFalse(ges.utils.rule_1(j,i,A))
            self.assertFalse(ges.utils.rule_2(i,j,A))
            self.assertFalse(ges.utils.rule_2(j,i,A))
            self.assertFalse(ges.utils.rule_4(i,j,A))
            self.assertFalse(ges.utils.rule_4(j,i,A))

    def test_meek_rule_3_b(self):
        A = np.array([[0,1,1,1],
                      [1,0,0,1],
                      [0,0,0,1],
                      [1,0,0,0]])
        # The edge 0 - 3 is not oriented by rule 3 but by rule 2
        self.assertFalse(ges.utils.rule_3(0,3,A))
        self.assertFalse(ges.utils.rule_3(3,0,A))
        self.assertTrue(ges.utils.rule_2(0,3,A))
        self.assertFalse(ges.utils.rule_2(3,0,A))
        # No rule should orient the 0 - 1 edge
        self.assertFalse(ges.utils.rule_1(0,1,A))
        self.assertFalse(ges.utils.rule_1(1,0,A))
        self.assertFalse(ges.utils.rule_2(0,1,A))
        self.assertFalse(ges.utils.rule_2(1,0,A))
        self.assertFalse(ges.utils.rule_3(0,1,A))
        self.assertFalse(ges.utils.rule_3(1,0,A))
        self.assertFalse(ges.utils.rule_4(0,1,A))
        self.assertFalse(ges.utils.rule_4(1,0,A))

    def test_meek_rule_3_c(self):
        A = np.array([[0,1,1,1,1,0],
                      [1,0,0,1,1,0],
                      [1,0,0,1,1,0],
                      [1,0,0,0,0,0],
                      [1,0,0,0,0,0],
                      [0,0,0,0,1,0]])
        # The edges 0 - 3  and 0 - 4 should be oriented by rule 3
        self.assertTrue(ges.utils.rule_3(0,3,A))
        self.assertFalse(ges.utils.rule_3(3,0,A))
        self.assertTrue(ges.utils.rule_3(0,4,A))
        self.assertFalse(ges.utils.rule_3(4,0,A))
        # The edges 0 - 1  and 0 - 2 should not be oriented by rule 3
        self.assertFalse(ges.utils.rule_3(0,1,A))
        self.assertFalse(ges.utils.rule_3(1,0,A))
        self.assertFalse(ges.utils.rule_3(0,2,A))
        self.assertFalse(ges.utils.rule_3(2,0,A))
        # The edge 4 - 0 should be oriented to 4 -> 0 by rule 1
        self.assertTrue(ges.utils.rule_1(4,0,A))
        self.assertFalse(ges.utils.rule_1(0,4,A))
        # No other rules should orient edges
        for i,j in [(0,3),(0,4),(0,1),(0,2)]:
            self.assertFalse(ges.utils.rule_2(i,j,A))
            self.assertFalse(ges.utils.rule_2(j,i,A))
            self.assertFalse(ges.utils.rule_4(i,j,A))
            self.assertFalse(ges.utils.rule_4(j,i,A))

    def test_meek_rule_4_a(self):
        A = np.array([[0,1,1,0],
                      [1,0,1,1],
                      [0,1,0,1],
                      [0,1,0,0]])
        self.assertTrue(ges.utils.rule_4(1,3,A))
        self.assertFalse(ges.utils.rule_4(3,1,A))
        # The other edges should not be oriented by rule 4
        self.assertFalse(ges.utils.rule_4(0,1,A))
        self.assertFalse(ges.utils.rule_4(1,0,A))
        self.assertFalse(ges.utils.rule_4(1,2,A))
        self.assertFalse(ges.utils.rule_4(2,1,A))
        # No rule should orient the other edges
        for i,j in [(1,3),(0,1),(1,2)]:
            self.assertFalse(ges.utils.rule_1(i,j,A))
            self.assertFalse(ges.utils.rule_1(j,i,A))
            self.assertFalse(ges.utils.rule_2(i,j,A))
            self.assertFalse(ges.utils.rule_2(j,i,A))
            self.assertFalse(ges.utils.rule_3(i,j,A))
            self.assertFalse(ges.utils.rule_3(j,i,A))

    def test_meek_rule_4_b(self):
        A = np.array([[0,1,0,0,1],
                      [0,0,1,0,1],
                      [0,0,0,1,1],
                      [0,0,0,0,1],
                      [1,1,1,1,0]])
        for i,j in [(4,2),(4,3)]:
            # Rule 4 should orient 4 -> 2, 4 -> 3
            self.assertTrue(ges.utils.rule_4(i,j,A))
            self.assertFalse(ges.utils.rule_4(j,i,A))
            # But other rules should not
            self.assertFalse(ges.utils.rule_1(i,j,A))
            self.assertFalse(ges.utils.rule_1(j,i,A))
            self.assertFalse(ges.utils.rule_2(i,j,A))
            self.assertFalse(ges.utils.rule_2(j,i,A))
            self.assertFalse(ges.utils.rule_3(i,j,A))
            self.assertFalse(ges.utils.rule_3(j,i,A))
        # But the remaining edges should not be oriented by any rule
        for i,j in [(0,4),(1,4)]:
            self.assertFalse(ges.utils.rule_1(i,j,A))
            self.assertFalse(ges.utils.rule_1(j,i,A))
            self.assertFalse(ges.utils.rule_2(i,j,A))
            self.assertFalse(ges.utils.rule_2(j,i,A))
            self.assertFalse(ges.utils.rule_3(i,j,A))
            self.assertFalse(ges.utils.rule_3(j,i,A))
            self.assertFalse(ges.utils.rule_4(i,j,A))
            self.assertFalse(ges.utils.rule_4(j,i,A))
