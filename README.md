# Greedy Equivalence Search (GES) Algorithm for Causal Discovery

This is a python implementation of the GES algorithm from the paper [*"Optimal Structure Identification With Greedy Search"*]((https://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf)) by David Maxwell Chickering.

You can install it via pip:

```bash
pip install ges
```

The implementation has been thoroughly tested (see [tests](#tests)), and has been written with an emphasis on readability and keeping a tiny dependency stack.

## When you should (and shouldn't) use this implementation

To the best of my knowledge, the only other public implementation of GES is in the R package [`pcalg`](https://www.rdocumentation.org/packages/pcalg/versions/2.7-1). It can be called from Python through an easy-to-use wrapper in the [Causal Discovery Toolbox](https://github.com/FenTechSolutions/CausalDiscoveryToolbox), but given its scope, this library contains many additional dependencies (including tensorflow) and still requires you to have `R`.

Thus, **this implementation might be for you if**:

- you want a dependency-light implementation (the only dependence outside the Standard Library is numpy), or
- you want to rewrite parts of GES for your own research, but you'd rather do it in Python. There is an emphasis on readability, and everything is thoroughly documented and properly referenced back to the GES/GIES papers.

**You should NOT use this implementation if:**

- you dont care about a large dependency stack,
- you have no interest in modifying GES itself, or
- you care about speed, as the `pcalg` implementation is highly optimized and is **very** fast.

## Running the algorithm

### Using the gaussian BIC score: `ges.fit_bic`

This is the variant which is normally found in the literature, and the one which was implemented in the original paper. It is made available under the function `ges.fit_bic`.

```python
ges.fit_bic(data, A0 = None, phases = ['forward', 'backward', 'turning'], debug = 0)
```

**Parameters**

- **data** (np.array): the matrix containing the observations of each variable (each column corresponds to a variable).
- **A0** (np.array, optional): the initial CPDAG on which GES will run, where where `A0[i,j] != 0` implies `i -> j` and `A[i,j] != 0 & A[j,i] != 0` implies `i - j`. Defaults to the empty graph.
- **phases** (`[{'forward', 'backward', 'turning'}*]`, optional): this controls which phases of the GES procedure are run, and in which order. Defaults to ['forward', 'backward', 'turning']. The turning phase was found by [Hauser & Bühlmann (2012)](https://www.jmlr.org/papers/volume13/hauser12a/hauser12a.pdf) to improve estimation performace, and is implemented here too.
- **debug** (int, optional): if larger than 0, debug are traces printed. Higher values correspond to increased verbosity.

**Returns**
- **estimate** (np.array): the adjacency matrix of the estimated CPDAG
- **total_score** (float): the score of the estimate

**Example**

### Using a custom score: `ges.fit`

```python
ges.fit(score_class, A0 = None, phases = ['forward', 'backward', 'turning'], debug = 0)
```

## Code Structure

All the modules can be found inside the `ges/` directory. These include:

  - `ges.ges` which is the main module with the calls to start GES, and contains the implementation of the insert, delete and turn operators.
  - `ges.utils` contains auxiliary functions and the logic to transform a PDAG into a CPDAG, used after each application of an operator.
  - `ges.scores` contains the modules with the score classes:
      - `ges.scores.decomposable_score` contains the base class for decomposable score classes (see that module for more details).
      - `ges.scores.gauss_obs_l0_pen` contains an implementation of the cached gaussian BIC score, as used in the original GES paper.
   - `ges.test` contains the modules with the unit tests and tests comparing against the algorithm's implementation in the 'pcalg' package.   

## Tests

All components come with unit tests to match, and some property-based tests. The output of the overall procedure has been checked against that of the [`pcalg`](https://www.rdocumentation.org/packages/pcalg/versions/2.7-1) implementation over tens of thousands of random graphs. Of course, this doesn't mean there are no bugs, but hopefully it means *they are less likely* :)

The tests can be run with `make test`. You can add `SUITE=<module_name>` to run a particular module only. There are, however, additional dependencies to run the tests. You can find these in [`requirements_tests.txt`](requirements_tests.txt).

The test modules are in the sub package `ges.test`, in the directory `ges/test`. They are

   - `ges.test.test_decomposable_score`: tests for decomposable score base class.
   - `ges.test.test_gauss_bic`: tests for the gaussian bic score.
   - `ges.test.test_operators`: tests for the insert, delete and turn operators.
   - `ges.test.test_pdag_to_cpdag`: tests the conversion from PDAG to CPDAG, which is applied after each application of an operator.
   - `ges.test.test_utils`: tests the other auxiliary functions    - `ges.test.test_vs_pcalg`: compares the output of the algorithm vs. that of `pcalg` for randomly generated graphs.

## Feedback

I hope you find this useful! Feedback and (constructive) criticism is always welcome, just shoot me an [email](mailto:juan.gamella@stat.math.ethz.ch) :)
