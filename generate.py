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


# ── Section 2: IR (Intermediate Representation) dataclasses ─────────────────
# These dataclasses represent a C function as a tree structure in memory,
# before it is rendered into C source text.
#
# Why a tree instead of strings?
#   Transformations (unrolling, inlining) need to inspect and mutate structure.
#   Operating on strings would require parsing them back, which is fragile.
#   A tree lets transformations walk and rebuild nodes cleanly.
#
# Upgrade path:
#   Statement is currently a plain string template. If future transformations
#   need to analyse arithmetic (e.g. strength reduction), Statement can be
#   expanded into an expression tree without changing anything above it.

from typing import Optional
from dataclasses import dataclass, field


@dataclass
class Statement:
    """A single C statement inside a loop body.

    `template` uses {var} as a placeholder for the loop variable so the
    renderer and unroller can substitute the correct index.

    Example:
        Statement("sum += arr[{var}];")
        -> rendered with var="i"  : "sum += arr[i];"
        -> unrolled with var="2"  : "sum += arr[2];"
    """
    template: str


@dataclass
class LoopSpec:
    """Describes one for-loop, including its body and optional inner loop.

    Nesting is represented recursively: a depth-3 loop is a LoopSpec whose
    inner_loop is a LoopSpec whose inner_loop is another LoopSpec.

    Attributes:
        variable:        loop variable name (i, j, k — one per depth level)
        iteration_count: fixed trip count; loop runs exactly this many times
        body:            statements executed each iteration (at this level only)
        inner_loop:      nested loop, or None if this is the innermost level
    """
    variable:        str
    iteration_count: int
    body:            list[Statement]     = field(default_factory=list)
    inner_loop:      Optional['LoopSpec'] = None


@dataclass
class ParameterSpec:
    """A single parameter in a C function signature.

    Example:
        ParameterSpec("arr", "int *") -> "int *arr"
        ParameterSpec("n",   "int")   -> "int n"
    """
    name:   str
    c_type: str


@dataclass
class FunctionSpec:
    """Top-level representation of a complete C function.

    This is the central object passed through the entire pipeline:
        FunctionGenerator creates it,
        transformations return modified copies of it,
        CRenderer turns it into a C source string.

    Attributes:
        name:        C function name, e.g. "func_0042"
        return_type: C return type, e.g. "int"
        parameters:  ordered list of function parameters
        loops:       top-level loops executed sequentially in the function body
        result_var:  name of the variable accumulated and returned
    """
    name:        str
    return_type: str
    parameters:  list[ParameterSpec] = field(default_factory=list)
    loops:       list[LoopSpec]      = field(default_factory=list)
    result_var:  str                 = "sum"  # name of the accumulator variable (e.g. sum += arr[i]); overridable per function
