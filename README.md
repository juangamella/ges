# Greedy Equivalence Search (GES) Algorithm for Causal Discovery

This is a python implementation of the GES algorithm from the paper [*"Optimal Structure Identification With Greedy Search"*]((https://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf)) by David Maxwell Chickering.

You can install it via pip:

```bash
pip install ges
```

The implementation has been thoroughly tested (see [tests](#tests)), and has been written with an emphasis on readability and keeping a tiny dependency stack.

## When you should use this implementation

To the best of my knowledge, the only other public implementation of GES is in the R package [`pcalg`](https://www.rdocumentation.org/packages/pcalg/versions/2.7-1). It can be called from Python through an easy-to-use wrapper in the [Causal Discovery Toolbox](https://github.com/FenTechSolutions/CausalDiscoveryToolbox), but this library contains many additional dependencies (including tensorflow) and still requires you to have `R`.

Thus, **this implementation might be for you if**:

- you want a dependency-light implementation (the only dependence outside the Standard Library is numpy), or
- you want to rewrite parts of GES for your own research, but you'd rather do it in Python. There is an emphasis on readability. The code is thoroughly documented and everything is properly referenced back to the GES/GIES papers.

**You should NOT use this implementation if:**

- You dont care about a large dependency stack,
- you have no interest in modifying GES itself, or
- you care about speed; the `pcalg` implementation is highly optimized and is **very** fast.

## Running the algorithm

### Using the gaussian BIC score: `ges.fit_bic`

This is the variant which is normally found in the literature, and the one which was implemented in the original paper. It is made available under the function `ges.fit_bic`, which takes the following parameters

**Parameters**

**Returns**

**Example**

### Using a custom score

## Code Structure

## Tests

All components come with unit and "property-based" tests to match. The output of the overall procedure has been checked against that of the [`pcalg`](https://www.rdocumentation.org/packages/pcalg/versions/2.7-1) implementation for tens of thousands of random graphs. Of course this doesn't mean there are no bugs, just that *they are less likely*.

The tests can be found in the sub package `ges.test`, and are divided into modules depending on the components they test. They can be run with the makefile included in this repository:

```shell
make test
```
. There are, however are additional dependencies to run the [tests](#tests)).



## Feedback
