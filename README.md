# Greedy Equivalence Search (GES) algorithm for Causal Discovery

This is a python implementation of the GES algorithm from the [paper](https://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf) *"Optimal Structure Identification With Greedy Search"* by David Maxwell Chickering.

### When you should use this implementation

To the best of my knowledge, the only other public implementation of GES is in the R package [`pcalg`](https://www.rdocumentation.org/packages/pcalg/versions/2.7-1). It can be called from python through a wrapper in the [Causal Discovery Toolbox](https://github.com/FenTechSolutions/CausalDiscoveryToolbox) a.k.a. `cdt`. However, `cdt` contains many additional dependencies (including tensorflow) and still requires you to have `R`.

This implementation is for you if:
- you want a dependency-light implementation (the only dependence outside the Standard Library is numpy).
- you want to rewrite parts of GES for your own research, but you don't know R. There is an emphasis on readability and the code is thoroughly documented and everything is properly referenced.

This implementation is not for you if:
- You dont care about a large dependency stack or modifying GES itself
- You care about speed; the `pcalg` implementation is highly optimized and is **very** fast

## Running the algorithm

## Code Structure

## Tests

## Feedback
