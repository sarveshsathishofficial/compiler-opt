#!/usr/bin/env bash
# benchmark.sh
# Compiles every generated C variant and measures its runtime.
#
# For each function in output/:
#   - Compiles original.c and unrolled.c with -O0 (no compiler optimizations)
#   - Runs each binary N_RUNS times and records all runtimes
#   - Writes the median runtime to results.csv
#
# Why -O0?
#   We want to measure the effect of our manual transformations.
#   Higher optimization levels let the compiler unroll/inline on its own,
#   which would make original and unrolled identical after compilation.
#
# Output: results.csv with columns:
#   func_name, transformation, median_ns

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────

OUTPUT_DIR="output"
RESULTS_FILE="results.csv"
N_RUNS=1000          # number of timed runs per binary — more runs = stable median
CC=gcc
CFLAGS="-O0"         # no compiler optimization — measure our transforms only
ARRAY_SIZE=64        # size of arr[] passed to each function

# ── Helpers ──────────────────────────────────────────────────────────────────

# time_binary <binary_path>
# Runs the binary N_RUNS times and prints the median runtime in nanoseconds.
time_binary() {
    local binary="$1"
    local times=()

    for ((i = 0; i < N_RUNS; i++)); do
        # clock_gettime via /usr/bin/time is too coarse; use perf stat or
        # a harness binary instead. Here we use a wrapper approach:
        # the binary itself prints its runtime (see harness template below).
        local ns
        ns=$("$binary")
        times+=("$ns")
    done

    # Sort and pick the median (index N_RUNS/2).
    local sorted
    sorted=$(printf '%s\n' "${times[@]}" | sort -n)
    local median_idx=$(( N_RUNS / 2 ))
    echo "$sorted" | sed -n "$((median_idx + 1))p"
}

# ── Main ─────────────────────────────────────────────────────────────────────

# Write CSV header.
echo "func_name,transformation,median_ns" > "$RESULTS_FILE"

total=$(ls -d "$OUTPUT_DIR"/func_* 2>/dev/null | wc -l)
count=0

for func_dir in "$OUTPUT_DIR"/func_*/; do
    func_name=$(basename "$func_dir")

    for c_file in "$func_dir"*.c; do
        transformation=$(basename "$c_file" .c)
        binary="${c_file%.c}"

        # ── Build a self-timing harness around the generated function ─────
        # Use Python to generate the harness file — avoids all bash string
        # escaping issues with special characters in C source (backslashes,
        # percent signs, quotes). Python writes the file verbatim with no
        # shell interpolation surprises.

        harness_file=$(mktemp /tmp/harness_XXXXXX.c)

        python3 - "$c_file" "$func_name" "$ARRAY_SIZE" "$harness_file" <<'PYEOF'
import sys

c_file, func_name, array_size, out_file = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]

with open(c_file) as f:
    c_source = f.read()

harness = f"""\
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

{c_source}

int main(void) {{
    int arr[{array_size}];
    for (int i = 0; i < {array_size}; i++) arr[i] = i + 1;
    volatile int sink = {func_name}(arr, {array_size});
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    sink = {func_name}(arr, {array_size});
    clock_gettime(CLOCK_MONOTONIC, &end);
    long ns = (end.tv_sec - start.tv_sec) * 1000000000L
            + (end.tv_nsec - start.tv_nsec);
    printf("%ld\\n", ns);
    return (int)(sink & 0);
}}
"""

with open(out_file, "w") as f:
    f.write(harness)
PYEOF

        # Compile and clean up temp file.
        $CC $CFLAGS "$harness_file" -o "$binary" -lm
        rm -f "$harness_file"

        # Time the binary and record the median.
        median=$(time_binary "$binary")
        echo "${func_name},${transformation},${median}" >> "$RESULTS_FILE"

        # Clean up the binary to avoid filling disk with 2000 executables.
        rm -f "$binary"
    done

    count=$(( count + 1 ))
    if (( count % 100 == 0 )); then
        echo "  $count/$total functions benchmarked"
    fi
done

echo "Done. Results written to $RESULTS_FILE"
