# extract.py
# Extracts structural features from each generated FunctionSpec and joins
# them with benchmark results to produce a labeled dataset for ML training.
#
# Why extract from the IR rather than parsing C files?
#   The FunctionSpec tree already contains everything we need. Parsing the
#   generated C would add a dependency (pycparser) and re-derive information
#   we already have. Re-running the generator with the same seed reproduces
#   identical specs deterministically.
#
# Output: dataset.csv
#   One row per function. Columns: structural features + label (unrolled_faster).
#
# Phase 2 note:
#   Current features describe loop structure only. Future phases should add
#   body-level features (data dependency type, memory access pattern, op count)
#   so the model generalises to unseen loop bodies.

# ── Imports ──────────────────────────────────────────────────────────────────

import csv
import random
from generate import (
    Config, FunctionGenerator, FunctionSpec, LoopSpec, Statement,
    MAX_UNROLLED_STMTS, LoopUnroller,
)


# ── Feature extraction ───────────────────────────────────────────────────────

def extract_features(spec: FunctionSpec) -> dict:
    """Extract structural features from a FunctionSpec.

    Returns a flat dict suitable for one CSV row.

    Current features (Phase 1 — loop structure only):
        loop_count            number of top-level loops
        total_loop_count      total loops including all nested levels
        max_nesting_depth     deepest nesting level across all loops
        max_iteration_count   highest single-loop trip count
        avg_iteration_count   mean trip count across all loops
        total_iter_product    product of all trip counts (proxy for unrolled size)
        has_dependent_body    1 if any loop body uses += or *= (data dependency)
        has_independent_body  1 if any loop body is an independent transformation
    """
    all_loops = _collect_all_loops(spec)

    iteration_counts = [l.iteration_count for l in all_loops]
    total_product    = 1
    for c in iteration_counts:
        total_product *= c

    # Detect body pattern types across all loops.
    all_stmts = [stmt for loop in all_loops for stmt in loop.body]
    has_dependent   = int(any("+=" in s.template or "*=" in s.template for s in all_stmts))
    has_independent = int(any("+=" not in s.template and "*=" not in s.template for s in all_stmts))

    return {
        "loop_count":           len(spec.loops),
        "total_loop_count":     len(all_loops),
        "max_nesting_depth":    _max_depth(spec),
        "max_iteration_count":  max(iteration_counts),
        "avg_iteration_count":  round(sum(iteration_counts) / len(iteration_counts), 2),
        "total_iter_product":   total_product,
        "has_dependent_body":   has_dependent,
        "has_independent_body": has_independent,
    }


def _collect_all_loops(spec: FunctionSpec) -> list[LoopSpec]:
    """Return a flat list of every loop in the spec, including nested ones."""
    result = []
    for loop in spec.loops:
        _collect_loops_recursive(loop, result)
    return result


def _collect_loops_recursive(loop: LoopSpec, result: list) -> None:
    """Recursively walk the loop tree and append every node to result."""
    result.append(loop)
    if loop.inner_loop is not None:
        _collect_loops_recursive(loop.inner_loop, result)


def _max_depth(spec: FunctionSpec) -> int:
    """Return the maximum nesting depth across all top-level loops."""
    return max(_loop_depth(loop) for loop in spec.loops)


def _loop_depth(loop: LoopSpec, current: int = 1) -> int:
    """Recursively compute nesting depth of a single loop chain."""
    if loop.inner_loop is None:
        return current
    return _loop_depth(loop.inner_loop, current + 1)


# ── Label loading ─────────────────────────────────────────────────────────────

def load_labels(results_path: str) -> dict[str, int]:
    """Load benchmark results and compute per-function label.

    Label = 1 if unrolled variant was strictly faster than original.
    Label = 0 if same speed or slower (not worth unrolling).

    Returns dict: func_name -> label
    """
    timings = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            timings.setdefault(row["func_name"], {})[row["transformation"]] = int(row["median_ns"])

    labels = {}
    for func_name, variants in timings.items():
        if "original" not in variants or "unrolled" not in variants:
            continue
        labels[func_name] = int(variants["unrolled"] < variants["original"])

    return labels


# ── Main ─────────────────────────────────────────────────────────────────────

RESULTS_PATH = "results.csv"
DATASET_PATH = "dataset.csv"

FEATURE_COLUMNS = [
    "loop_count",
    "total_loop_count",
    "max_nesting_depth",
    "max_iteration_count",
    "avg_iteration_count",
    "total_iter_product",
    "has_dependent_body",
    "has_independent_body",
]


def main() -> None:
    config = Config()
    rng    = random.Random(config.seed)
    gen    = FunctionGenerator(config, rng)

    print(f"Loading labels from {RESULTS_PATH}...")
    labels = load_labels(RESULTS_PATH)

    rows = []
    for i in range(config.n_functions):
        func_name = f"func_{i:04d}"
        spec      = gen.generate(i)
        features  = extract_features(spec)
        label     = labels.get(func_name)

        if label is None:
            # No benchmark result for this function — skip it.
            continue

        row = {"func_name": func_name, **features, "unrolled_faster": label}
        rows.append(row)

    # Write dataset.csv
    fieldnames = ["func_name"] + FEATURE_COLUMNS + ["unrolled_faster"]
    with open(DATASET_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {DATASET_PATH}")

    # Print a quick feature summary so the user can sanity-check the output.
    _print_summary(rows)


def _print_summary(rows: list[dict]) -> None:
    """Print basic stats on the extracted features and label distribution."""
    total   = len(rows)
    pos     = sum(r["unrolled_faster"] for r in rows)
    neg     = total - pos

    print(f"\nLabel distribution:")
    print(f"  Unrolled faster (1): {pos} ({100*pos//total}%)")
    print(f"  Not faster      (0): {neg} ({100*neg//total}%)")

    print(f"\nFeature ranges:")
    for col in FEATURE_COLUMNS:
        values = [r[col] for r in rows]
        print(f"  {col:<25} min={min(values):<6} max={max(values):<8} avg={sum(values)/len(values):.1f}")


if __name__ == "__main__":
    main()
