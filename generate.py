# generate.py
# Generates synthetic C functions and their transformed variants for use as
# an ML training dataset. Each function is written to disk in both its
# original form and one or more optimized forms (e.g. unrolled loops).
#
# Pipeline:
#   Config -> FunctionGenerator -> FunctionSpec (IR)
#          -> TransformationRegistry -> transformed specs
#          -> CRenderer -> C source strings
#          -> OutputWriter -> files on disk

# ── Section 1: Config ───────────────────────────────────────────────────────
# All tunable knobs live here. Nothing else in the file should contain
# magic numbers the user would want to change.

from dataclasses import dataclass, field


# Top-level constants — change these to scale the dataset.
N_FUNCTIONS           = 1000
SEED                  = 42        # fixed seed for reproducibility
OUTPUT_DIR            = "output"
ITERATION_COUNT_RANGE = (2, 16)   # how many times each loop runs
LOOP_COUNT_RANGE      = (1, 3)    # how many top-level loops per function
NESTING_DEPTH_RANGE   = (1, 3)    # 1 = flat, 2 = one level of nesting, 3 = two levels of nesting

# Which variants to write for every function.
# "original" is always the identity (no transformation).
# Add new transformation names here once they are registered in main().
ENABLED_TRANSFORMATIONS = ["original", "unrolled"]


@dataclass
class Config:
    """Single source of truth for all generation parameters.

    Passed to every component so nothing reads module-level globals directly.
    This makes the components independently testable and the whole pipeline
    easy to run with different settings without editing constants.
    """
    n_functions:             int   = N_FUNCTIONS
    seed:                    int   = SEED
    output_dir:              str   = OUTPUT_DIR
    iteration_count_range:   tuple = ITERATION_COUNT_RANGE
    loop_count_range:        tuple = LOOP_COUNT_RANGE
    nesting_depth_range:     tuple = NESTING_DEPTH_RANGE   # max depth of 3 avoids combinatorial explosion while covering real-world nesting patterns
    enabled_transformations: list  = field(
        default_factory=lambda: list(ENABLED_TRANSFORMATIONS)
    )
