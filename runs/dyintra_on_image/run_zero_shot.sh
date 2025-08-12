#!/bin/bash

MODEL_PATH=/local/scratch/bmg44/dual_stream_runs/checkpoints/dyintra_on_image/run_20250527_195129/

CHECKPOINTS=(1107020)

EVAL_DIR=evaluation_data/full_eval

for checkpoint in "${CHECKPOINTS[@]}"; do
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name ${MODEL_PATH}checkpoint_${checkpoint}.pt --backend dst --task blimp --data_path "${EVAL_DIR}/blimp_filtered" --save_predictions > runs/dyintra_on_image/blimp_${checkpoint}.txt
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name ${MODEL_PATH}checkpoint_${checkpoint}.pt --backend dst --task blimp --data_path "${EVAL_DIR}/supplement_filtered" --save_predictions > runs/dyintra_on_image/blimp_supplement_${checkpoint}.txt
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name ${MODEL_PATH}checkpoint_${checkpoint}.pt --backend dst --task ewok --data_path "${EVAL_DIR}/ewok_filtered" --save_predictions > runs/dyintra_on_image/ewok_${checkpoint}.txt
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name ${MODEL_PATH}checkpoint_${checkpoint}.pt --backend dst --task entity_tracking --data_path "${EVAL_DIR}/entity_tracking" --save_predictions > runs/dyintra_on_image/entity_tracking_${checkpoint}.txt
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name ${MODEL_PATH}checkpoint_${checkpoint}.pt --backend dst --task wug_adj --data_path "${EVAL_DIR}/wug_adj_nominalization" --save_predictions > runs/dyintra_on_image/wug_adj_nominalization_${checkpoint}.txt
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name ${MODEL_PATH}checkpoint_${checkpoint}.pt --backend dst --task wug_past --data_path "${EVAL_DIR}/wug_past_tense" --save_predictions > runs/dyintra_on_image/wug_past_tense_${checkpoint}.txt
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name ${MODEL_PATH}checkpoint_${checkpoint}.pt --backend dst --task comps --data_path "${EVAL_DIR}/comps" --save_predictions > runs/dyintra_on_image/comps_${checkpoint}.txt
    python -m evaluation_pipeline.reading.run --model_path_or_name ${MODEL_PATH}checkpoint_${checkpoint}.pt --backend dst --data_path "${EVAL_DIR}/reading/reading_data.csv" > runs/dyintra_on_image/reading_${checkpoint}.txt
done
