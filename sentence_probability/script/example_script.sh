#!/bin/bash

set +e
set -x

# ============================================================
# Paths
# ============================================================

CONSTRUCTION_DATA="./data/20261001construction_data_expansion_preprocessed.xlsx"
PROMPT_DATA="./data/prompts.xlsx"

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/all_models_${RUN_TIMESTAMP}.log"

# Save both stdout and stderr to the log while still printing to terminal.
exec > >(tee -a "$LOG_FILE") 2>&1


# ============================================================
# Hugging Face cache
#
# Priority:
#   1. HF_HUB_CACHE, if explicitly set
#   2. $HF_HOME/hub, if HF_HOME is set
#   3. ~/.cache/huggingface/hub
# ============================================================

if [ -n "${HF_HUB_CACHE:-}" ]; then
    HF_HUB_DIR="$HF_HUB_CACHE"
elif [ -n "${HF_HOME:-}" ]; then
    HF_HUB_DIR="$HF_HOME/hub"
else
    HF_HUB_DIR="$HOME/.cache/huggingface/hub"
fi

echo "Hugging Face hub cache directory: $HF_HUB_DIR"


# ============================================================
# Models
#
# Keep Aya Expanse 32B LAST because it is the heaviest model.
# ============================================================

MODELS=(
    "openai/gpt-oss-120b"
)


# ============================================================
# Entire-script timer
# ============================================================

TOTAL_START_TIME=$(date +%s)
TOTAL_START_DATE=$(date)

echo "============================================================"
echo "All-model run started at: $TOTAL_START_DATE"
echo "Log file: $LOG_FILE"
echo "Number of models: ${#MODELS[@]}"
echo "HF hub cache: $HF_HUB_DIR"
echo "============================================================"


# ============================================================
# Run models sequentially
# ============================================================

for MODEL in "${MODELS[@]}"; do

    # Convert Hugging Face path to output/model name convention.
    #
    # Example:
    #   CohereLabs/aya-expanse-8b
    #       ->
    #   CohereLabs_aya-expanse-8b
    #
    MODEL_NAME="${MODEL//\//_}"

    MODEL_START_TIME=$(date +%s)
    MODEL_START_DATE=$(date)

    echo ""
    echo "============================================================"
    echo "Starting model: $MODEL"
    echo "Model name:     $MODEL_NAME"
    echo "Started at:     $MODEL_START_DATE"
    echo "============================================================"

    # --------------------------------------------------------
    # Run sentence_probability2.py
    # --------------------------------------------------------

    python -u ./src/sentence_probability2.py \
        --construction_data "$CONSTRUCTION_DATA" \
        --prompt_data "$PROMPT_DATA" \
        --model "$MODEL" \
        --model_name "$MODEL_NAME" \
        --final_exp

    EXIT_CODE=$?

    MODEL_END_TIME=$(date +%s)
    MODEL_END_DATE=$(date)

    MODEL_ELAPSED=$((MODEL_END_TIME - MODEL_START_TIME))
    MODEL_HOURS=$((MODEL_ELAPSED / 3600))
    MODEL_MINUTES=$(((MODEL_ELAPSED % 3600) / 60))
    MODEL_SECONDS=$((MODEL_ELAPSED % 60))

    echo ""
    echo "------------------------------------------------------------"
    echo "Finished model: $MODEL"
    echo "Finished at:    $MODEL_END_DATE"
    echo "Exit code:      $EXIT_CODE"
    echo "Runtime:        ${MODEL_HOURS}h ${MODEL_MINUTES}m ${MODEL_SECONDS}s"
    echo "------------------------------------------------------------"


    # ========================================================
    # Delete this model from the Hugging Face cache
    #
    # Only delete when the model completed successfully.
    #
    # Hugging Face cache convention:
    #
    #   CohereLabs/aya-expanse-8b
    #
    # becomes:
    #
    #   models--CohereLabs--aya-expanse-8b
    #
    # If the model failed, keep the cache so that a retry does
    # not require downloading the entire model again.
    # ========================================================

    if [ "$EXIT_CODE" -eq 0 ]; then

        MODEL_CACHE_NAME="models--${MODEL//\//--}"
        MODEL_CACHE_PATH="$HF_HUB_DIR/$MODEL_CACHE_NAME"

        echo ""
        echo "============================================================"
        echo "Model completed successfully."
        echo "Cleaning Hugging Face cache."
        echo "Model:      $MODEL"
        echo "Cache path: $MODEL_CACHE_PATH"
        echo "============================================================"

        if [ -d "$MODEL_CACHE_PATH" ]; then

            echo ""
            echo "Cache size before deletion:"
            du -sh "$MODEL_CACHE_PATH" || true

            echo ""
            echo "Deleting cached model:"
            echo "$MODEL_CACHE_PATH"

            rm -rf -- "$MODEL_CACHE_PATH"

            echo ""

            if [ -d "$MODEL_CACHE_PATH" ]; then
                echo "WARNING: Hugging Face cache deletion failed."
                echo "Directory still exists:"
                echo "$MODEL_CACHE_PATH"
            else
                echo "Successfully deleted Hugging Face model cache:"
                echo "$MODEL"
            fi

        else

            echo ""
            echo "No Hugging Face cache directory found for this model."
            echo "Nothing to delete."

        fi

    else

        echo ""
        echo "============================================================"
        echo "WARNING: Model failed."
        echo "Model:     $MODEL"
        echo "Exit code: $EXIT_CODE"
        echo ""
        echo "Keeping its Hugging Face cache for debugging/retry."
        echo "============================================================"

    fi


    # ========================================================
    # Disk-space status after each model
    # ========================================================

    echo ""
    echo "Disk usage after model:"
    df -h "$HOME" || true

    echo ""
    echo "============================================================"
    echo "Completed processing for: $MODEL"
    echo "Moving to next model."
    echo "============================================================"

done


# ============================================================
# Entire-script runtime
# ============================================================

TOTAL_END_TIME=$(date +%s)
TOTAL_END_DATE=$(date)

TOTAL_ELAPSED=$((TOTAL_END_TIME - TOTAL_START_TIME))
TOTAL_HOURS=$((TOTAL_ELAPSED / 3600))
TOTAL_MINUTES=$(((TOTAL_ELAPSED % 3600) / 60))
TOTAL_SECONDS=$((TOTAL_ELAPSED % 60))

echo ""
echo "============================================================"
echo "ALL MODELS FINISHED"
echo "Started at:    $TOTAL_START_DATE"
echo "Finished at:   $TOTAL_END_DATE"
echo "Total runtime: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo "Log file:      $LOG_FILE"
echo "============================================================"