#!/bin/bash

MODEL_PATH=/local/scratch/bmg44/dual_stream_runs/checkpoints/base/run_20250508_141536/

CHECKPOINTS=(100000 150000 200000 250000 300000 350000 400000 450000 500000 550000 600000 650000 700000 750000 800000 850000 900000 950000 1000000 1050000 1100000 1107020)

EVAL_DIR=evaluation_data/full_eval

for checkpoint in "${CHECKPOINTS[@]}"; do
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name ${MODEL_PATH}checkpoint_${checkpoint}.pt --backend dst --task blimp --data_path "${EVAL_DIR}/blimp_filtered" --save_predictions > runs/base/blimp_${checkpoint}.txt
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name ${MODEL_PATH}checkpoint_${checkpoint}.pt --backend dst --task blimp --data_path "${EVAL_DIR}/supplement_filtered" --save_predictions > runs/base/blimp_supplement_${checkpoint}.txt
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name ${MODEL_PATH}checkpoint_${checkpoint}.pt --backend dst --task ewok --data_path "${EVAL_DIR}/ewok_filtered" --save_predictions > runs/base/ewok_${checkpoint}.txt
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name ${MODEL_PATH}checkpoint_${checkpoint}.pt --backend dst --task entity_tracking --data_path "${EVAL_DIR}/entity_tracking" --save_predictions > runs/base/entity_tracking_${checkpoint}.txt
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name ${MODEL_PATH}checkpoint_${checkpoint}.pt --backend dst --task wug_adj --data_path "${EVAL_DIR}/wug_adj_nominalization" --save_predictions > runs/base/wug_adj_nominalization_${checkpoint}.txt
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name ${MODEL_PATH}checkpoint_${checkpoint}.pt --backend dst --task wug_past --data_path "${EVAL_DIR}/wug_past_tense" --save_predictions > runs/base/wug_past_tense_${checkpoint}.txt
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name ${MODEL_PATH}checkpoint_${checkpoint}.pt --backend dst --task comps --data_path "${EVAL_DIR}/comps" --save_predictions > runs/base/comps_${checkpoint}.txt
done
