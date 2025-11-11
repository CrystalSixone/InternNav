# export HF_ENDPOINT=https://hf-mirror.com

# Usage: bash download_split.sh /path/to/log_file.log
# Parse the log for missing scenes and download corresponding .tar files from HF

LOG_FILE="${1:-}"
DATASET_REPO="billzhao1030/all_scene"
TARGET_DIR="/cpfs/shared/simulation/wangliuyi/vlnverse_scene"

if [ -z "$LOG_FILE" ] || [ ! -f "$LOG_FILE" ]; then
    echo "Usage: bash download_split.sh /path/to/log_file.log"
    exit 1
fi

mkdir -p "$TARGET_DIR"

# Extract unique scene ids like kujiale_0123 or kujiale_0123_fix from log
mapfile -t SCENE_IDS < <(grep -oE 'kujiale_[0-9]{4}(_fix)?' "$LOG_FILE" | sort -u)

if [ "${#SCENE_IDS[@]}" -eq 0 ]; then
    echo "No missing scene ids found in log: $LOG_FILE"
    exit 0
fi

echo "Found ${#SCENE_IDS[@]} scene ids in log. Filtering out already-downloaded .tar files..."

INCLUDES_ARGS=()
SKIPPED_COUNT=0
for id in "${SCENE_IDS[@]}"; do
    tar_path="$TARGET_DIR/${id}.tar"
    if [ -f "$tar_path" ]; then
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        continue
    fi
    INCLUDES_ARGS+=(--include "${id}.tar")
done

if [ "${#INCLUDES_ARGS[@]}" -eq 0 ]; then
    echo "All listed scene .tar files already exist locally ($SKIPPED_COUNT skipped). Nothing to download."
    exit 0
fi

echo "Will download ${#INCLUDES_ARGS[@]} .tar files (skipped $SKIPPED_COUNT existing)."

# Some environments of huggingface-cli handle multiple --include poorly.
# Download sequentially to ensure every .tar is fetched.
DOWNLOADED_COUNT=0
for id in "${SCENE_IDS[@]}"; do
    tar_path="$TARGET_DIR/${id}.tar"
    if [ -f "$tar_path" ]; then
        continue
    fi
    echo "Downloading ${id}.tar ..."
    if huggingface-cli download "$DATASET_REPO" \
        --repo-type dataset \
        --local-dir "$TARGET_DIR" \
        --resume-download \
        --include "${id}.tar"; then
        DOWNLOADED_COUNT=$((DOWNLOADED_COUNT + 1))
    else
        echo "Failed to download ${id}.tar" >&2
    fi
done

echo "Download finished: ${DOWNLOADED_COUNT} new, ${SKIPPED_COUNT} skipped. Files saved to $TARGET_DIR."