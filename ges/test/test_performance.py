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

# ---------------------------------------------------------------------

NUM_GRAPHS = 2
NUM_RUNS = 10
DEBUG = False


class GES_speed_tests(unittest.TestCase):

    # ------------------------------------------------------
    # Tests

    def test_1(self):
        p = 15  # number of variables
        n = 1500  # size of the observational sample
        times = np.zeros((NUM_GRAPHS, NUM_RUNS), dtype=float)
        samples = np.zeros((NUM_GRAPHS, n, p), dtype=float)
        Ws = np.zeros((NUM_GRAPHS, p, p))
        for r in range(NUM_RUNS):
            print("Run %d/%d" % (r + 1, NUM_RUNS)) if DEBUG else None
            for i in range(NUM_GRAPHS):
                print("  Checking SCM %d/%d" %
                      (i+1, NUM_GRAPHS)) if DEBUG else None
                start = time.time()
                W = sempler.generators.dag_avg_deg(
                    p, 3, 0.5, 1, random_state=i)
                scm = sempler.LGANM(W, (1, 10), (0.5, 1), random_state=i)
                obs_sample = scm.sample(n=n, random_state=i)
                # Make sure sample is the same
                if r == 0:
                    Ws[i, :, :] = W
                    samples[i, :, :] = obs_sample
                else:
                    assert (W == Ws[i]).all()
                    assert (obs_sample == samples[i]).all()
                # Run GES
                start = time.time()
                estimate, _ = ges.fit_bic(
                    obs_sample, iterate=True, debug=0)
                end = time.time()
                times[i, r] = end - start
        # ------------------
        # Display statistics
        print("Stats. per graph")
        for i in range(NUM_GRAPHS):
            print("  i = %d - avg. = %0.4f - std. %0.4f" %
                  (i, times[i].mean(), times[i].std()))
        print("\n Total stats")
        print("  avg. = %0.4f - std. %0.4f" % (times.mean(), times.std()))
