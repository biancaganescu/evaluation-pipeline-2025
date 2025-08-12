#!/bin/bash

MODEL_PATH=/local/scratch/bmg44/dual_stream_runs/checkpoints/medium_size/run_20250521_215700/
CHECKPOINTS=(100000 200000 300000 400000 500000 600000 700000 800000 900000 1000000 1100000 1107020)

EVAL_DIR=evaluation_data/full_eval

for checkpoint in "${CHECKPOINTS[@]}"; do
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name ${MODEL_PATH}checkpoint_${checkpoint}.pt --backend dst --task blimp --data_path "${EVAL_DIR}/blimp_filtered" --save_predictions > runs/data_text_first/blimp_${checkpoint}.txt
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name ${MODEL_PATH}checkpoint_${checkpoint}.pt --backend dst --task blimp --data_path "${EVAL_DIR}/supplement_filtered" --save_predictions > runs/data_text_first/blimp_supplement_${checkpoint}.txt
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name ${MODEL_PATH}checkpoint_${checkpoint}.pt --backend dst --task ewok --data_path "${EVAL_DIR}/ewok_filtered" --save_predictions > runs/data_text_first/ewok_${checkpoint}.txt
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name ${MODEL_PATH}checkpoint_${checkpoint}.pt --backend dst --task entity_tracking --data_path "${EVAL_DIR}/entity_tracking" --save_predictions > runs/data_text_first/entity_tracking_${checkpoint}.txt
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name ${MODEL_PATH}checkpoint_${checkpoint}.pt --backend dst --task wug_adj --data_path "${EVAL_DIR}/wug_adj_nominalization" --save_predictions > runs/data_text_first/wug_adj_nominalization_${checkpoint}.txt
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name ${MODEL_PATH}checkpoint_${checkpoint}.pt --backend dst --task wug_past --data_path "${EVAL_DIR}/wug_past_tense" --save_predictions > runs/data_text_first/wug_past_tense_${checkpoint}.txt
    python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name ${MODEL_PATH}checkpoint_${checkpoint}.pt --backend dst --task comps --data_path "${EVAL_DIR}/comps" --save_predictions > runs/data_text_first/comps_${checkpoint}.txt
    python -m evaluation_pipeline.reading.run --model_path_or_name ${MODEL_PATH}checkpoint_${checkpoint}.pt --backend dst --data_path "${EVAL_DIR}/reading/reading_data.csv" > runs/data_text_first/reading_${checkpoint}.txt
done
