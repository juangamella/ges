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
Test vs. the output of the R implementation of GES.
"""

import unittest
import numpy as np
import sempler
import sempler.generators
import time

import ges
import ges.scores.gauss_obs_l0_pen
import ges.utils
import gies.utils
import gnies.utils

# ---------------------------------------------------------------------

NUM_GRAPHS = 100
NUM_RUNS = 10
DEBUG = True


class EnumerationTests(unittest.TestCase):

    # ------------------------------------------------------
    # Tests

    def test_1(self):
        p = 15  # number of variables
        times = np.zeros((3, NUM_GRAPHS, NUM_RUNS), dtype=float)
        As = np.zeros((NUM_GRAPHS, p, p))
        for r in range(NUM_RUNS):
            print("Run %d/%d" % (r + 1, NUM_RUNS)) if DEBUG else None
            for i in range(NUM_GRAPHS):
                print("  Checking graph %d/%d" %
                      (i+1, NUM_GRAPHS)) if DEBUG else None
                A = sempler.generators.dag_avg_deg(
                    p, 1, 1, 1, random_state=i)
                if r == 0:
                    As[i, :, :] = A
                else:
                    assert (A == As[i, :, :]).all()
                targets = [[0], [1], [2]]
                union = set.union(*[set(t) for t in targets])
                # CPDAG completion
                start = time.time()
                ges.utils.pdag_to_cpdag(A)
                end = time.time()
                times[0, i, r] = end - start
                # do-I-CPDAG completion
                start = time.time()
                gies.utils.replace_unprotected(A, targets)
                end = time.time()
                times[1, i, r] = end - start
                # nI-CPDAG completion
                start = time.time()
                gnies.utils.pdag_to_icpdag(A, union)
                end = time.time()
                times[2, i, r] = end - start
        # ------------------
        # Display statistics
        print("Stats. per graph")
        print("    CPDAG            do-ICPDAG         nI-CPDAG")
        for i in range(NUM_GRAPHS):
            avgs = times[:,i,:].mean(axis=1)
            stds = times[:,i,:].std(axis=1)
            print("%0.4f (%0.4f) - %0.4f (%0.4f) - %0.4f (%0.4f)"
                  % (avgs[0], stds[0], avgs[1], stds[1], avgs[2], stds[2]))
        print("\n Total stats")
        print(times.mean(axis=(1, 2)))
        print(times.std(axis=(1, 2)))
