#!/bin/bash

# Sync data from the 002_ts_resampling_strategies_isolated project.
# Usage: ./sync_data.sh [SOURCE_PROJECT_ROOT]
# Default SOURCE_PROJECT_ROOT: /Users/joaopms/Documents/002_ts_resampling_strategies_isolated

set -e

# This script lives AT the repo root, so its own directory is the repo root.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source paths
SOURCE_PROJECT="${1:-/Users/joaopms/Documents/002_ts_resampling_strategies_isolated}"
SERIES_SRC="$SOURCE_PROJECT/data/paper_datasets_csv"
RAW_ITER_SRC="$SOURCE_PROJECT/Results (Clean)/Results Data/raw_iterations_by_dataset_v2"

# Check source exists
if [[ ! -d "$SERIES_SRC" ]]; then
  echo "Error: Series source not found at $SERIES_SRC" >&2
  exit 1
fi
if [[ ! -d "$RAW_ITER_SRC" ]]; then
  echo "Error: Raw iterations source not found at $RAW_ITER_SRC" >&2
  exit 1
fi

# Destination paths
SERIES_DST="$REPO_ROOT/data/series"
RAW_ITER_DST="$REPO_ROOT/data/raw_iterations"

# Create directories
mkdir -p "$SERIES_DST" "$RAW_ITER_DST"

# Copy series (all files: CSVs, manifest, README, etc.)
cp -f "$SERIES_SRC"/* "$SERIES_DST/"

# Copy raw iterations (all CSV files that exist)
cp -f "$RAW_ITER_SRC"/*.csv "$RAW_ITER_DST/"

# Regenerate SNAPSHOT.md with current checksums and discovered dataset IDs
SNAPSHOT_FILE="$REPO_ROOT/data/SNAPSHOT.md"
SNAP_DATE=$(date +%Y-%m-%d)

# Discover dataset IDs by globbing, sorted
series_ids=$(cd "$SERIES_DST" && ls -1 DS*.csv 2>/dev/null | sed -E 's/DS([0-9]*).*/\1/' | sort -n | sed 's/^/DS/')
raw_iter_ids=$(cd "$RAW_ITER_DST" && ls -1 DS*.csv 2>/dev/null | sed -E 's/DS([0-9]*).*/\1/' | sort -n | sed 's/^/DS/')

# Datasets with a raw series but no per-fold results yet. Derived by diffing
# the two discovered ID lists -- never a hardcoded DS01..DS24 sweep, which is
# exactly the dataset-count assumption PLAN.md's Staging section forbids.
missing_ids=$(comm -23 <(echo "$series_ids") <(echo "$raw_iter_ids") | tr '\n' ',' | sed 's/,$//')

# Count files
series_count=$(cd "$SERIES_DST" && ls -1 | wc -l)
# `|| true`: grep exits 1 when every file is a DS*.csv, which set -e would
# otherwise treat as a sync failure.
other_files=$(cd "$SERIES_DST" && ls -1 | grep -v '^DS.*\.csv$' | tr '\n' ',' | sed 's/,$//' || true)
raw_iter_count=$(cd "$RAW_ITER_DST" && ls -1 DS*.csv 2>/dev/null | wc -l)

# Generate checksums
series_checksums=$(cd "$SERIES_DST" && shasum -a 256 * | sort)
raw_iter_checksums=$(cd "$RAW_ITER_DST" && shasum -a 256 DS*.csv 2>/dev/null | sort)

# Format dataset ID lists with proper display
series_ids_formatted=$(echo "$series_ids" | tr '\n' ', ' | sed 's/,$//')
raw_iter_ids_formatted=$(echo "$raw_iter_ids" | tr '\n' ', ' | sed 's/,$//')

# Create SNAPSHOT.md
cat > "$SNAPSHOT_FILE" <<EOF
# Data Vendoring Snapshot

**Snapshot Date:** $SNAP_DATE

## Source Paths

- **Series Data Source:** \`$SERIES_SRC\`
- **Raw Iterations Source:** \`$RAW_ITER_SRC\`

## Dataset Inventory

### data/series/ ($series_count files)

**Datasets Present:** $series_ids_formatted

**Other Files:** $other_files

**SHA-256 Checksums:**

\`\`\`
$series_checksums
\`\`\`

### data/raw_iterations/ ($raw_iter_count files)

**Datasets Present:** $raw_iter_ids_formatted

$(
if [[ -n "$missing_ids" ]]; then
  echo "**Datasets Missing:** $missing_ids"
else
  echo "**All datasets present.**"
fi
)

**SHA-256 Checksums:**

\`\`\`
$raw_iter_checksums
\`\`\`

## Notes

- All files copied verbatim from source paths
- To refresh data later, run \`./sync_data.sh\` with source project root as argument
EOF

echo "✓ Data sync complete"
echo "  - Series: $series_count files"
echo "  - Raw iterations: $raw_iter_count files"
echo "  - SNAPSHOT.md regenerated"
