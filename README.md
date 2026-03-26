# compiler-opt

A dataset generator and benchmarking pipeline for training ML models to predict
which compiler optimizations will improve performance on a given piece of C code.

## What it does

1. **Generates** synthetic C functions with random loop structures, nesting depths, and body patterns
2. **Transforms** each function into optimized variants (currently: loop unrolling)
3. **Benchmarks** each variant by compiling and timing it
4. **Produces** a labeled dataset (`results.csv`) of code features and measured runtimes for ML training

The goal is to teach a model to answer: *given this code, which transformation will make it run faster?*

## Background

Compilers like GCC and LLVM use hand-written heuristics to decide when to apply
optimizations like loop unrolling and function inlining. These rules are tuned for
specific hardware and degrade over time. This project explores whether an ML model
trained on measured runtime data can make better decisions than those hand-written rules.

## Project structure

```
generate.py      — generates synthetic C functions and their transformed variants
benchmark.sh     — compiles and times each variant, writes results.csv
output/          — generated C files (git-ignored, recreate with generate.py)
results.csv      — benchmark output (git-ignored, recreate with benchmark.sh)
```

## Pipeline

```
generate.py  →  output/func_NNNN/{original,unrolled}.c
benchmark.sh →  results.csv (func_name, transformation, median_ns)
```

## Usage

### 1. Generate functions

```bash
python3 generate.py
```

Produces `output/func_0000/` through `output/func_0999/`, each containing one `.c` file
per enabled transformation.

### 2. Benchmark

```bash
bash benchmark.sh
```

Compiles every variant with `-O0` (no compiler optimization) and measures runtime
using `clock_gettime(CLOCK_MONOTONIC)`. Results written to `results.csv`.

### 3. Configure

Edit the constants at the top of `generate.py`:

| Constant | Default | Description |
|---|---|---|
| `N_FUNCTIONS` | 1000 | Number of functions to generate |
| `SEED` | 42 | Random seed for reproducibility |
| `ITERATION_COUNT_RANGE` | (2, 16) | How many times each loop runs |
| `LOOP_COUNT_RANGE` | (1, 3) | Top-level loops per function |
| `NESTING_DEPTH_RANGE` | (1, 3) | Max loop nesting depth |
| `ENABLED_TRANSFORMATIONS` | original, unrolled | Which variants to generate |

## Extending

### Adding a new transformation

1. Write a class with a `transform(spec: FunctionSpec) -> FunctionSpec` method in `generate.py`
2. Register it in `main()`: `registry.register("my_transform", MyTransform().transform)`
3. Add `"my_transform"` to `ENABLED_TRANSFORMATIONS`

The rest of the pipeline (rendering, writing, benchmarking) handles it automatically.

## Roadmap

- **Phase 1** (current): unrolling on synthetic functions with fixed body patterns
- **Phase 2**: add inlining and dead code elimination transformations
- **Phase 3**: extract richer AST features (data dependencies, memory access patterns)
- **Phase 4**: train and evaluate ML model (random forest → XGBoost)
- **Phase 5**: token sequence features + neural network

## Requirements

- Python 3.10+
- GCC
- bash
