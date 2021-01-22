# Greedy Equivalence Search (GES) algorithm for Causal Discovery

This is a python implementation of the GES algorithm from the [paper](https://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf) *"Optimal Structure Identification With Greedy Search"* by David Maxwell Chickering. The implementation has a very small dependency stack, as the only dependency outside the Standard Library is numpy. You can install it via pip:

```bash
pip install ges
```
The code has been thoroughly tested via unit testing and its output compared against that of the R package [`pcalg`] for thousands of random graphs
(note that there are additional dependencies to run the [tests](#tests)).

## When you should use this implementation

To the best of my knowledge, the only other public implementation of GES is in the R package [`pcalg`](https://www.rdocumentation.org/packages/pcalg/versions/2.7-1). It can be called from python through a wrapper in the [Causal Discovery Toolbox](https://github.com/FenTechSolutions/CausalDiscoveryToolbox) a.k.a. `cdt`. However, `cdt` contains many additional dependencies (including tensorflow) and still requires you to have `R`.

Thus, **this implementation might be for you if**:

- you want a dependency-light implementation (the only dependence outside the Standard Library is numpy), or
- you want to rewrite parts of GES for your own research, but you'd rather do it in Python. There is an emphasis on readability. The code is thoroughly documented and everything is properly referenced back to the GES/GIES papers.

You should NOT use this implementation if:

- You dont care about a large dependency stack,
- you have no interest in modifying GES itself, or
- you care about speed; the `pcalg` implementation is highly optimized and is **very** fast.

## Running the algorithm

## Code Structure

## Tests

## Feedback
