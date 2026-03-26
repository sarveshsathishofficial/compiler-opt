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
        # The generated .c file contains only the function definition.
        # We compile it together with an inline main() that:
        #   1. initialises arr[] with dummy data
        #   2. calls the function inside a timed loop
        #   3. prints the elapsed nanoseconds
        #
        # volatile sink prevents the compiler from optimising away the call.

        harness=$(cat <<HARNESS
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

$(cat "$c_file")

int main(void) {
    int arr[$ARRAY_SIZE];
    /* fill array with non-zero data so operations are meaningful */
    for (int i = 0; i < $ARRAY_SIZE; i++) arr[i] = i + 1;

    /* warm up: run once before timing to prime caches */
    volatile int sink = ${func_name}(arr, $ARRAY_SIZE);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    sink = ${func_name}(arr, $ARRAY_SIZE);

    clock_gettime(CLOCK_MONOTONIC, &end);

    long ns = (end.tv_sec - start.tv_sec) * 1000000000L
            + (end.tv_nsec - start.tv_nsec);
    printf("%ld\n", ns);

    /* use sink so the compiler cannot eliminate the function call */
    return (int)(sink & 0);
}
HARNESS
)
        # Compile the harness to a temporary binary.
        echo "$harness" | $CC $CFLAGS -x c - -o "$binary" -lm

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
