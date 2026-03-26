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
ITERATION_COUNT_RANGE = (2, 64)   # wider range: short loops favour unrolling, long loops hurt code cache
MAX_UNROLLED_STMTS    = 5000      # hard cap: if unrolling would exceed this many statements, keep the loop
LOOP_COUNT_RANGE      = (1, 3)    # how many top-level loops per function
NESTING_DEPTH_RANGE   = (1, 5)    # deeper nesting (up to 4 levels) produces large unrolled code, balancing the dataset

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
    max_unrolled_stmts:      int   = MAX_UNROLLED_STMTS


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


# Loop variable names assigned by nesting depth (depth 0 = i, 1 = j, ... 4 = m).
_LOOP_VARS = ["i", "j", "k", "l", "m"]


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
        max_depth = self._config.nesting_depth_range[1] - 1  # 0-indexed

        # Scale the iteration count down as nesting gets deeper.
        # Without this, a depth-5 loop with 128 iterations produces 128^5
        # statements when unrolled, exhausting memory.
        # Each depth level halves the max allowed iterations.
        base_max  = self._config.iteration_count_range[1]
        scaled_max = max(2, base_max // (2 ** depth))
        iteration_count = self._rng.randint(self._config.iteration_count_range[0], scaled_max)

        variable  = _LOOP_VARS[depth]

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

    def transform(self, spec: FunctionSpec, max_stmts: int = MAX_UNROLLED_STMTS) -> FunctionSpec:
        """Return a new FunctionSpec with all loops fully unrolled.

        If the estimated total unrolled statement count across all loops would
        exceed max_stmts, the original spec is returned unchanged. This prevents
        pathologically large files from being generated for deeply nested loops
        with high iteration counts — which would exhaust memory and crash.

        Returning the original unchanged is valid data for the ML model: it means
        "this function was too complex to unroll" and the label will be "no benefit"
        since original == unrolled in that case.
        """
        # Estimate total statements before doing any work.
        estimated = sum(self._estimate_stmts(loop) for loop in spec.loops)
        if estimated > max_stmts:
            return spec  # too large — return original unchanged

        new_spec       = copy.deepcopy(spec)
        new_spec.loops = [self._unroll_loop(loop) for loop in new_spec.loops]
        return new_spec

    def _estimate_stmts(self, loop: LoopSpec) -> int:
        """Estimate how many statements unrolling this loop would produce.

        Multiplies iteration counts down the nesting chain recursively.
        Used as a cheap pre-check before actually unrolling.
        """
        body_count = len(loop.body)
        if loop.inner_loop is not None:
            inner_count = self._estimate_stmts(loop.inner_loop)
            return loop.iteration_count * (body_count + inner_count)
        return loop.iteration_count * max(body_count, 1)

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


# ── Section 5: CRenderer ─────────────────────────────────────────────────────
# Turns a FunctionSpec into a C source string.
# Knows C syntax and indentation. Knows nothing about files, transformations,
# or random generation. Stateless — same spec always produces the same text.

INDENT = "    "  # 4 spaces per indentation level


class CRenderer:
    """Renders a FunctionSpec as a valid C source string.

    Handles:
      - Function signature with parameters
      - Result variable declaration and return
      - Nested for-loops at arbitrary depth
      - Unrolled loops (iteration_count=1, no inner_loop) emitted as flat statements
    """

    def render(self, spec: FunctionSpec) -> str:
        """Return a complete C source string for the given FunctionSpec."""
        lines = []

        # -- function signature --
        params = ", ".join(f"{p.c_type} {p.name}" for p in spec.parameters)
        lines.append(f"{spec.return_type} {spec.name}({params}) {{")

        # -- declare and zero the accumulator --
        lines.append(f"{INDENT}int {spec.result_var} = 0;")
        lines.append("")

        # -- render each top-level loop sequentially --
        for loop in spec.loops:
            loop_lines = self._render_loop(loop, depth=1)
            lines.extend(loop_lines)
            lines.append("")

        # -- return the accumulated result --
        lines.append(f"{INDENT}return {spec.result_var};")
        lines.append("}")

        return "\n".join(lines)

    def _render_loop(self, loop: LoopSpec, depth: int) -> list[str]:
        """Recursively render a loop and its body at the given indentation depth.

        If the loop has iteration_count=1 and no inner_loop, it was produced by
        LoopUnroller and should be emitted as flat statements (no for-wrapper).
        Otherwise it is rendered as a standard for-loop.
        """
        indent = INDENT * depth
        lines  = []

        is_unrolled = (loop.iteration_count == 1 and loop.inner_loop is None)

        if is_unrolled:
            # Emit the pre-expanded statements directly, no for-loop wrapper.
            for stmt in loop.body:
                lines.append(f"{indent}{stmt.template}")
        else:
            # Emit a standard for-loop.
            var   = loop.variable
            count = loop.iteration_count
            lines.append(f"{indent}for (int {var} = 0; {var} < {count}; {var}++) {{")

            # Render body statements, substituting {var} with the loop variable name.
            for stmt in loop.body:
                rendered = stmt.template.replace(f"{{{var}}}", var)
                lines.append(f"{indent}{INDENT}{rendered}")

            # Render the inner loop recursively if present.
            if loop.inner_loop is not None:
                inner_lines = self._render_loop(loop.inner_loop, depth + 1)
                lines.extend(inner_lines)

            lines.append(f"{indent}}}")

        return lines


# ── Section 6: OutputWriter ──────────────────────────────────────────────────
# Owns all file system interaction. Renders a spec and writes it to disk.
# No transformation logic, no random logic — only path construction and I/O.
#
# Output structure:
#   output/
#     func_0000/
#       original.c
#       unrolled.c
#     func_0001/
#       ...
#
# One directory per function. Each transformation is one .c file inside it.
# This layout makes it easy to load all variants of a function together during
# ML training without any path gymnastics.

import os


class OutputWriter:
    """Writes rendered C source files to disk under the configured output directory."""

    def __init__(self, config: Config, renderer: CRenderer) -> None:
        self._config   = config
        self._renderer = renderer

    def write(self, spec: FunctionSpec, transformation_name: str) -> str:
        """Render spec and write it to output/<func_name>/<transformation_name>.c.

        Returns the path of the written file.
        """
        dir_path  = os.path.join(self._config.output_dir, spec.name)
        file_path = os.path.join(dir_path, f"{transformation_name}.c")

        os.makedirs(dir_path, exist_ok=True)

        source = self._renderer.render(spec)
        with open(file_path, "w") as f:
            f.write(source)

        return file_path


# ── Section 7: TransformationRegistry ───────────────────────────────────────
# Maps transformation names to callables.
# The registry is the single extension point for adding new transformations.
# The main loop iterates over enabled transformations by name, looks each up
# here, and applies it — without knowing anything about the transformation itself.

from typing import Callable


class TransformationRegistry:
    """Maps transformation names to transform(FunctionSpec) -> FunctionSpec callables.

    Usage:
        registry = TransformationRegistry()
        registry.register("original", lambda s: s)
        registry.register("unrolled", LoopUnroller().transform)

        for name, fn in registry.get_enabled(config):
            transformed = fn(original_spec)
    """

    def __init__(self) -> None:
        # Internal dict: name -> callable
        self._transforms: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None:
        """Register a transformation under a given name."""
        self._transforms[name] = fn

    def get_enabled(self, config: Config) -> list[tuple[str, Callable]]:
        """Return (name, fn) pairs for all transformations listed in config.

        Raises KeyError if a name in config.enabled_transformations has not
        been registered — fail loudly rather than silently skipping.
        """
        result = []
        for name in config.enabled_transformations:
            if name not in self._transforms:
                raise KeyError(
                    f"Transformation '{name}' is in ENABLED_TRANSFORMATIONS "
                    f"but was never registered. Register it in main()."
                )
            result.append((name, self._transforms[name]))
        return result


# ── Section 8: main ──────────────────────────────────────────────────────────
# Wires all components together and runs the generation pipeline.
# This is the only place that knows about all components simultaneously.

import sys


def main() -> None:
    config   = Config()
    rng      = _random.Random(config.seed)
    renderer = CRenderer()
    writer   = OutputWriter(config, renderer)
    gen      = FunctionGenerator(config, rng)

    # Register transformations.
    # "original" is always the identity — no changes, just renders as-is.
    # Add new transformations here as the project grows.
    registry = TransformationRegistry()
    registry.register("original", lambda s: s)
    registry.register("unrolled", LoopUnroller().transform)

    enabled = registry.get_enabled(config)

    print(f"Generating {config.n_functions} functions -> {config.output_dir}/")

    for i in range(config.n_functions):
        spec = gen.generate(index=i)

        for name, transform_fn in enabled:
            transformed = transform_fn(spec)
            writer.write(transformed, transformation_name=name)

        # Print progress every 100 functions so the user knows it's running.
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{config.n_functions}")

    print("Done.")


if __name__ == "__main__":
    main()
