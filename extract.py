# extract.py
# Extracts structural and body-level features from each generated FunctionSpec
# and joins them with benchmark results to produce a labeled dataset for ML training.
#
# Why extract from the IR rather than parsing C files?
#   The FunctionSpec tree already contains everything we need. Parsing the
#   generated C would add a dependency (pycparser) and re-derive information
#   we already have. Re-running the generator with the same seed reproduces
#   identical specs deterministically.
#
# Label (multi-class):
#   0 = original is fastest
#   1 = unrolled is fastest
#   2 = inlined is fastest
#   3 = unrolled_inlined is fastest
#
# Output: dataset.csv
#   One row per function. Columns: features + best_transformation label.

# ── Imports ──────────────────────────────────────────────────────────────────

import csv
import random
from generate import (
    Config, FunctionGenerator, FunctionSpec, LoopSpec, Statement, HelperSpec,
    MAX_UNROLLED_STMTS, LoopUnroller,
)

# Maps transformation name to integer class label.
LABEL_MAP = {
    "original":        0,
    "unrolled":        1,
    "inlined":         2,
    "unrolled_inlined": 3,
}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


# ── Feature extraction ───────────────────────────────────────────────────────

def extract_features(spec: FunctionSpec) -> dict:
    """Extract structural, body-level, and inlining features from a FunctionSpec.

    Returns a flat dict suitable for one CSV row.

    Phase 1 — loop structure:
        loop_count            number of top-level loops
        total_loop_count      total loops including all nested levels
        max_nesting_depth     deepest nesting level across all loops
        max_iteration_count   highest single-loop trip count
        avg_iteration_count   mean trip count across all loops
        total_iter_product    product of all trip counts (proxy for unrolled size)

    Phase 2 — body level:
        has_reduction         1 if any stmt accumulates into a scalar (data dependency)
        array_reads_per_iter  avg array reads per iteration
        array_writes_per_iter avg array writes per iteration
        has_multiply          1 if any body contains multiplication
        total_body_stmts      total statements across all loop bodies

    Phase 2 — inlining features:
        has_function_call     1 if any loop body calls a helper function
        num_helpers           number of distinct helper functions defined
        helper_body_ops       total operator count across all helper body expressions
                              — proxy for how expensive inlining each call is
        call_count            total number of helper call statements in all loop bodies
    """
    all_loops = _collect_all_loops(spec)

    iteration_counts = [l.iteration_count for l in all_loops]
    total_product    = 1
    for c in iteration_counts:
        total_product *= c

    all_stmts = [stmt for loop in all_loops for stmt in loop.body]

    # ── Body-level features ───────────────────────────────────────────────────

    has_reduction = int(any(
        ("+=" in s.template or "*=" in s.template)
        and not s.template.strip().startswith("arr")
        for s in all_stmts
    ))

    reads_per_stmt       = [s.template.count("arr[") for s in all_stmts]
    array_reads_per_iter = round(
        sum(reads_per_stmt) / len(reads_per_stmt) if reads_per_stmt else 0, 2
    )

    writes_per_stmt       = [int(s.template.strip().startswith("arr")) for s in all_stmts]
    array_writes_per_iter = round(
        sum(writes_per_stmt) / len(writes_per_stmt) if writes_per_stmt else 0, 2
    )

    has_multiply     = int(any("*" in s.template for s in all_stmts))
    total_body_stmts = len(all_stmts)

    # ── Inlining features ─────────────────────────────────────────────────────

    # has_function_call: any loop body statement calls a helper.
    # Detected by presence of "(" in statement template (all calls use parens).
    has_function_call = int(any("(" in s.template for s in all_stmts))

    # num_helpers: number of distinct helper functions in this spec.
    num_helpers = len(spec.helpers)

    # helper_body_ops: count operators (+, -, *, /, >>, <<) in each helper body.
    # More operators = more work inlined per call site = more potential benefit.
    ops = set("+-*/<>")
    helper_body_ops = sum(
        sum(1 for ch in h.body_expr if ch in ops)
        for h in spec.helpers
    )

    # call_count: how many statements across all loops are helper calls.
    call_count = sum(1 for s in all_stmts if "(" in s.template)

    return {
        # Phase 1 — loop structure
        "loop_count":            len(spec.loops),
        "total_loop_count":      len(all_loops),
        "max_nesting_depth":     _max_depth(spec),
        "max_iteration_count":   max(iteration_counts),
        "avg_iteration_count":   round(sum(iteration_counts) / len(iteration_counts), 2),
        "total_iter_product":    total_product,
        # Phase 2 — body level
        "has_reduction":         has_reduction,
        "array_reads_per_iter":  array_reads_per_iter,
        "array_writes_per_iter": array_writes_per_iter,
        "has_multiply":          has_multiply,
        "total_body_stmts":      total_body_stmts,
        # Phase 2 — inlining
        "has_function_call":     has_function_call,
        "num_helpers":           num_helpers,
        "helper_body_ops":       helper_body_ops,
        "call_count":            call_count,
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

# All transformation variants we expect in results.csv.
ALL_TRANSFORMATIONS = ["original", "unrolled", "inlined", "unrolled_inlined"]


def load_labels(results_path: str) -> dict[str, int]:
    """Load benchmark results and compute the best transformation per function.

    Label = integer class of whichever variant had the lowest median runtime:
        0 = original, 1 = unrolled, 2 = inlined, 3 = unrolled_inlined

    Ties are broken by preferring the simpler transformation (lower label index)
    to avoid rewarding unnecessary complexity.

    Returns dict: func_name -> label
    """
    timings = {}
    with open(results_path) as f:
        for row in csv.DictReader(f):
            timings.setdefault(row["func_name"], {})[row["transformation"]] = int(row["median_ns"])

    labels = {}
    for func_name, variants in timings.items():
        # Only label functions that have all 4 variants measured.
        if not all(t in variants for t in ALL_TRANSFORMATIONS):
            continue

        # Find the transformation with the lowest runtime.
        best_name = min(ALL_TRANSFORMATIONS, key=lambda t: variants[t])
        labels[func_name] = LABEL_MAP[best_name]

    return labels


# ── Main ─────────────────────────────────────────────────────────────────────

RESULTS_PATH = "results.csv"
DATASET_PATH = "dataset.csv"

FEATURE_COLUMNS = [
    # Phase 1 — loop structure
    "loop_count",
    "total_loop_count",
    "max_nesting_depth",
    "max_iteration_count",
    "avg_iteration_count",
    "total_iter_product",
    # Phase 2 — body level
    "has_reduction",
    "array_reads_per_iter",
    "array_writes_per_iter",
    "has_multiply",
    "total_body_stmts",
    # Phase 2 — inlining
    "has_function_call",
    "num_helpers",
    "helper_body_ops",
    "call_count",
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
            continue

        row = {"func_name": func_name, **features, "best_transformation": label}
        rows.append(row)

    fieldnames = ["func_name"] + FEATURE_COLUMNS + ["best_transformation"]
    with open(DATASET_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {DATASET_PATH}")
    _print_summary(rows)


def _print_summary(rows: list[dict]) -> None:
    """Print label distribution and feature ranges."""
    total  = len(rows)
    counts = {i: 0 for i in range(4)}
    for r in rows:
        counts[r["best_transformation"]] += 1

    print(f"\nLabel distribution (best transformation):")
    for label_int, name in LABEL_NAMES.items():
        n = counts[label_int]
        print(f"  {label_int} {name:<20} {n:>5} ({100*n//total}%)")

    print(f"\nFeature ranges:")
    for col in FEATURE_COLUMNS:
        values = [r[col] for r in rows]
        print(f"  {col:<25} min={min(values):<6} max={max(values):<8} avg={sum(values)/len(values):.1f}")


if __name__ == "__main__":
    main()
