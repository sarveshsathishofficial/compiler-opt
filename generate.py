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

# Body patterns used inside loop iterations.
# {var} is replaced with the loop variable at render/unroll time.
# These cover the most common loop body shapes:
#   - dependent reductions (each iteration needs the previous result)
#   - independent transformations (iterations can run in parallel)
# Phase 2 will replace these with feature extraction so the model generalises
# to loop bodies it was not explicitly trained on.
BODY_PATTERNS = [
    "sum += arr[{var}];",              # addition reduction (dependent)
    "sum *= arr[{var}];",              # multiplication reduction (dependent)
    "arr[{var}] = arr[{var}] * 2;",   # independent in-place transformation
    "sum += arr[{var}] * arr[{var}];", # multiply-accumulate (dependent)
]


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
    body_patterns:           list  = field(
        default_factory=lambda: list(BODY_PATTERNS)
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


# ── Section 3: FunctionGenerator ────────────────────────────────────────────
# Responsible for creating random FunctionSpec instances.
# All random decisions are made here and nowhere else.
# Takes a seeded random.Random instance so output is reproducible and
# safe to parallelize later without global state interference.

import random as _random


# Loop variable names assigned by nesting depth (depth 0 = i, 1 = j, 2 = k).
_LOOP_VARS = ["i", "j", "k"]


class FunctionGenerator:
    """Creates random FunctionSpec instances from a Config.

    Usage:
        rng = random.Random(config.seed)
        gen = FunctionGenerator(config, rng)
        spec = gen.generate(index=0)
    """

    def __init__(self, config: Config, rng: _random.Random) -> None:
        self._config = config
        self._rng    = rng

    def generate(self, index: int) -> FunctionSpec:
        """Generate one function spec with a unique name based on index."""
        name       = f"func_{index:04d}"
        loop_count = self._rng.randint(*self._config.loop_count_range)
        loops      = [self._make_loop(depth=0) for _ in range(loop_count)]

        return FunctionSpec(
            name        = name,
            return_type = "int",
            parameters  = [
                ParameterSpec("arr", "int *"),
                ParameterSpec("n",   "int"),
            ],
            loops      = loops,
            result_var = "sum",
        )

    def _make_loop(self, depth: int) -> LoopSpec:
        """Recursively build a loop, nesting down to a random max depth.

        depth=0 is the outermost loop. Each call may create an inner loop
        until the configured max nesting depth is reached.
        """
        max_depth       = self._config.nesting_depth_range[1] - 1  # 0-indexed
        iteration_count = self._rng.randint(*self._config.iteration_count_range)
        variable        = _LOOP_VARS[depth]

        # Pick a random body pattern for this loop level.
        # The pattern is chosen once per loop so all iterations share it.
        pattern = self._rng.choice(self._config.body_patterns)

        # At the innermost level, add the actual work statement.
        # At outer levels, the body is empty — work happens in the inner loop.
        if depth >= max_depth:
            body       = [Statement(pattern.replace("{var}", f"{{{variable}}}"))]
            inner_loop = None
        else:
            # Randomly decide whether to nest deeper or stop here.
            go_deeper  = self._rng.random() < 0.5
            if go_deeper:
                body       = []
                inner_loop = self._make_loop(depth + 1)
            else:
                body       = [Statement(pattern.replace("{var}", f"{{{variable}}}"))]
                inner_loop = None

        return LoopSpec(
            variable        = variable,
            iteration_count = iteration_count,
            body            = body,
            inner_loop      = inner_loop,
        )


# ── Section 4: Transformations ───────────────────────────────────────────────
# Each transformation is a class with a single transform() method:
#     transform(spec: FunctionSpec) -> FunctionSpec
#
# Transformations must never mutate the input spec — always return a new one.
# This keeps the original available for the "original.c" render and makes
# transformations safe to compose in any order.
#
# Adding a new transformation (e.g. inlining, dead code elimination):
#   1. Write a new class with a transform() method below.
#   2. Register it in main() via registry.register("name", MyTransform().transform).
#   3. Add "name" to ENABLED_TRANSFORMATIONS in Config.
#   Nothing else needs to change.

import copy


class LoopUnroller:
    """Unrolls every loop in a FunctionSpec by replacing it with explicit statements.

    Unrolling removes loop overhead (counter increment, condition check, branch)
    by writing out each iteration as a separate statement.

    Example — original loop (iteration_count=3, variable="i"):
        for (int i = 0; i < 3; i++) { sum += arr[i]; }

    After unrolling:
        sum += arr[0];
        sum += arr[1];
        sum += arr[2];

    Nested loops are unrolled from the inside out: the innermost loop is
    expanded first, then its parent, preserving correct iteration order.
    """

    def transform(self, spec: FunctionSpec) -> FunctionSpec:
        """Return a new FunctionSpec with all loops fully unrolled."""
        new_spec       = copy.deepcopy(spec)
        new_spec.loops = [self._unroll_loop(loop) for loop in new_spec.loops]
        return new_spec

    def _unroll_loop(self, loop: LoopSpec) -> LoopSpec:
        """Recursively unroll a loop and any inner loops.

        Returns a LoopSpec with iteration_count=1 whose body contains all
        the expanded statements. iteration_count=1 signals to the renderer
        that this loop has been unrolled and should be emitted without a
        for-loop wrapper.
        """
        if loop.inner_loop is not None:
            # Unroll the inner loop first, then unroll this level around it.
            unrolled_inner = self._unroll_loop(loop.inner_loop)
            inner_stmts    = self._expand(unrolled_inner)
        else:
            inner_stmts = []

        # Expand this loop's own body statements across all iterations.
        expanded_body = self._expand_statements(loop.body, loop.variable, loop.iteration_count)

        # Combine: for each iteration of this loop, emit body then inner stmts.
        all_stmts = []
        for i in range(loop.iteration_count):
            # Substitute the current iteration index into body statements.
            for stmt in loop.body:
                substituted = stmt.template.replace(f"{{{loop.variable}}}", str(i))
                all_stmts.append(Statement(substituted))
            # Append inner loop statements (already fully expanded).
            all_stmts.extend(inner_stmts)

        # Return a degenerate LoopSpec (iteration_count=1) carrying flat statements.
        # The renderer checks iteration_count=1 + no inner_loop to skip the for-wrapper.
        return LoopSpec(
            variable        = loop.variable,
            iteration_count = 1,
            body            = all_stmts,
            inner_loop      = None,
        )

    def _expand(self, loop: LoopSpec) -> list[Statement]:
        """Flatten a fully-unrolled LoopSpec into a plain list of statements."""
        return loop.body

    def _expand_statements(
        self,
        stmts: list[Statement],
        variable: str,
        count: int,
    ) -> list[Statement]:
        """Substitute variable with each index 0..count-1 across all statements.

        Not used directly in _unroll_loop but available for future transformations
        that need to expand a statement list independently.
        """
        result = []
        for i in range(count):
            for stmt in stmts:
                substituted = stmt.template.replace(f"{{{variable}}}", str(i))
                result.append(Statement(substituted))
        return result
