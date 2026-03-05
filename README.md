# Lunaris MoC

> **Experimental branch notice**
>
> This branch contains ongoing experimental work on the **MoC (Mixture-of-Collaboration)** and **IRL (Iterative Reasoning Loop)** architecture, as well as updates to the training pipeline. These changes are still under active development and have **not** yet been thoroughly validated through extensive testing or long-run experiments.

## Status

The code in this branch should be considered **experimental**.

While it includes recent bug fixes, refactors, and new features, the current implementation of the architecture and training system is still being iterated on. Behavior, performance, stability, and training characteristics may change as development continues.

## Recommendation

For users who want the most stable and better-tested version of the project, the recommended choice is the **`main`** branch.

The `main` branch contains the version that has already been used in previous experiments and is currently the best reference point for reproducible runs and baseline comparisons.

## Why this branch exists

This branch is primarily intended for:

- prototyping and testing new ideas in the MoC architecture;
- iterating on IRL-related mechanisms;
- integrating bug fixes and implementation changes before they are promoted to `main`;
- validating new training-system features and experimental behavior.

## Important note

The presence of bug fixes or newer features in this branch does **not** mean that it is the recommended branch for general use.

Until the changes here are more thoroughly benchmarked and validated, **`main` remains the recommended branch for training, evaluation, and general experimentation**.

## Project focus

The central focus of this project is the development of the **MoC** and **IRL** modules as core architectural components.

This branch reflects active research and engineering work around those modules, but it should be treated as a work in progress rather than a finalized or production-ready implementation.

---

If you are exploring the project for the first time, start with **`main`**.
If you specifically want to inspect the latest experimental changes, then this branch is the appropriate place to look.
