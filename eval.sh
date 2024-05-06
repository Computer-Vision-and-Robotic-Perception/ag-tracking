#!/bin/bash

source venv/bin/activate

echo Evaluating AppleMOT...
python TrackEval/scripts/run_mot_challenge.py   --GT_FOLDER TrackEval/data/AppleMOT/sequences/testing \
                                                --TRACKERS_FOLDER TrackEval/data/trackers_apple \
                                                --BENCHMARK AppleMOT \
                                                --METRICS HOTA CLEAR Identity\
                                                --SPLIT_TO_EVAL test \
                                                --TRACKER_SUB_FOLDER ""\
                                                --OUTPUT_FOLDER TrackEval/data/results_apple \
                                                --SEQMAP_FOLDER TrackEval/data/seqmaps \
                                                --SKIP_SPLIT_FOL True \
                                                --USE_PARALLEL True \
                                                --NUM_PARALLEL_CORES 16 \
                                                --PRINT_CONFIG False \
                                                --PRINT_RESULTS True \
                                                --OUTPUT_SUMMARY True \
                                                --OUTPUT_EMPTY_CLASSES True \
                                                --OUTPUT_DETAILED True \
                                                --PLOT_CURVES True > eval_apple.log

echo Evaluating LettuceMOT...
python TrackEval/scripts/run_mot_challenge.py   --GT_FOLDER TrackEval/data/LettuceMOT \
                                                --TRACKERS_FOLDER TrackEval/data/trackers_lettuce \
                                                --BENCHMARK LettuceMOT \
                                                --METRICS HOTA CLEAR Identity\
                                                --SPLIT_TO_EVAL all \
                                                --TRACKER_SUB_FOLDER ""\
                                                --OUTPUT_FOLDER TrackEval/data/results_lettuce \
                                                --SEQMAP_FOLDER TrackEval/data/seqmaps \
                                                --SKIP_SPLIT_FOL True \
                                                --USE_PARALLEL True \
                                                --NUM_PARALLEL_CORES 16 \
                                                --PRINT_CONFIG False \
                                                --PRINT_RESULTS True\
                                                --OUTPUT_SUMMARY True \
                                                --OUTPUT_EMPTY_CLASSES True \
                                                --OUTPUT_DETAILED True \
                                                --PLOT_CURVES True > eval_lettuce.log
