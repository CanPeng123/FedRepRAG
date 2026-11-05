#!/bin/bash
#SBATCH --mem=160G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100-pcie-80gb:1
#SBATCH -p gpu_long
#SBATCH --output=logs/%x_%j.txt

# ─────────────────────────────────────────────────────────────────────────────
# Usage (non-interactive, direct):
#   bash run.sh --config configs/slake.yaml --stage incontext
#   bash run.sh --config configs/kilt_nq.yaml --stage mlp_infer \
#               --retrieval_mlp_ckpt output/KILT_NQ/mlp_train_centralized/mlp_stage2_epoch010.pt
#
# Usage (interactive menu — no arguments needed):
#   bash run.sh
#
# Usage (SLURM):
#   sbatch --job-name=slake_mlp run.sh --config configs/slake.yaml --stage fl_mlp_train
# ─────────────────────────────────────────────────────────────────────────────

export TOKENIZERS_PARALLELISM=false
# export HF_TOKEN="your_huggingface_token"   # required for gated models (e.g. Gemma)

CODE_PTH="${CODE_PTH:-$(git rev-parse --show-toplevel 2>/dev/null)}"
export PYTHONPATH="${CODE_PTH}/src:${PYTHONPATH:-}"
mkdir -p "${CODE_PTH}/logs"

# ── If args are supplied, pass them straight through and exit ─────────────────
if [ $# -gt 0 ]; then
    python run.py --code_pth "${CODE_PTH}" "$@"
    exit $?
fi

# ── Interactive menu ──────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║          FedRepRAG Experiment Runner     ║"
echo "╚══════════════════════════════════════════╝"

# ── Step 1: choose dataset ────────────────────────────────────────────────────
echo ""
echo "Select dataset:"
echo "  1) SLAKE        (medical VQA)"
echo "  2) ScienceQA    (science VQA)"
echo "  3) PubMedQA       (medical QA)"
echo "  4) KILT-NQ      (open-domain QA)"
echo ""
read -rp "Enter number [1-4]: " DS_CHOICE

case "$DS_CHOICE" in
    1) CONFIG="configs/slake.yaml";     DATASET="SLAKE"    ;;
    2) CONFIG="configs/scienceqa.yaml"; DATASET="ScienceQA" ;;
    3) CONFIG="configs/pubmedqa.yaml";    DATASET="PubMedQA"   ;;
    4) CONFIG="configs/kilt_nq.yaml";   DATASET="KILT-NQ"  ;;
    *) echo "Invalid choice. Exiting."; exit 1 ;;
esac

# ── Step 2: choose FL or centralized ─────────────────────────────────────────
echo ""
echo "Select setting:"
echo "  1) Centralized"
echo "  2) Federated (FL)"
echo ""
read -rp "Enter number [1-2]: " FL_CHOICE

case "$FL_CHOICE" in
    1) FL_PREFIX=""    ;;
    2) FL_PREFIX="fl_" ;;
    *) echo "Invalid choice. Exiting."; exit 1 ;;
esac

# ── Step 3: choose experiment mode ───────────────────────────────────────────
echo ""
echo "Select experiment mode:"
echo "  1) Without RAG        (baseline, no retrieval)"
echo "  2) Get embeddings     (build FAISS retrieval index)"
echo "  3) In-context RAG     (retrieved examples in prompt)"
echo "  4) MLP training       (train retrieval projector)"
echo "  5) MLP inference      (run inference with trained projector)"
echo ""
read -rp "Enter number [1-5]: " MODE_CHOICE

case "$MODE_CHOICE" in
    1) STAGE="no_rag" ;;
    2)
        if [ "${FL_PREFIX}" = "fl_" ]; then
            if [ "$DATASET" = "PubMedQA" ]; then
                STAGE="get_fl_embedding"
            else
                STAGE="fl_get_embedding"
            fi
        else
            STAGE="get_embedding"
        fi
        ;;
    3)
        if [ "${FL_PREFIX}" = "fl_" ]; then
            STAGE="fl_incontext"
        else
            STAGE="incontext"
        fi
        ;;
    4) STAGE="${FL_PREFIX}mlp_train"  ;;
    5) STAGE="${FL_PREFIX}mlp_infer"  ;;
    *) echo "Invalid choice. Exiting."; exit 1 ;;
esac

# ── Step 4: checkpoint prompt for inference stages ────────────────────────────
EXTRA_ARGS=()
if [ "$MODE_CHOICE" = "5" ]; then
    echo ""
    echo "MLP inference requires a trained checkpoint."

    CKPT_ARG="checkpoint"

    read -rp "Enter checkpoint path (--${CKPT_ARG}): " CKPT_PATH
    if [ -z "$CKPT_PATH" ]; then
        echo "No checkpoint provided. Exiting."; exit 1
    fi
    EXTRA_ARGS=("--${CKPT_ARG}" "${CKPT_PATH}")
fi

# ── Summary and confirm ───────────────────────────────────────────────────────
echo ""
echo "─────────────────────────────────────────"
echo " Dataset : ${DATASET}"
echo " Config  : ${CONFIG}"
echo " Stage   : ${STAGE}"
[ "${#EXTRA_ARGS[@]}" -gt 0 ] && echo " Extra   : ${EXTRA_ARGS[*]}"
echo "─────────────────────────────────────────"
echo ""
read -rp "Run? [Y/n]: " CONFIRM
if [[ "$CONFIRM" =~ ^[Nn] ]]; then
    echo "Aborted."; exit 0
fi

echo ""
python run.py --code_pth "${CODE_PTH}" \
    --config "${CONFIG}" \
    --stage  "${STAGE}" \
    "${EXTRA_ARGS[@]}"