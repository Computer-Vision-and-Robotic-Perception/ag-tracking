#!/bin/bash

source venv/bin/activate
python TrackEval/scripts/run_mot_challenge.py   --GT_FOLDER TrackEval/data/LettuceMOT \
                                                --TRACKERS_FOLDER TrackEval/data/trackers \
                                                --BENCHMARK LettuceMOT \
                                                --METRICS HOTA CLEAR Identity\
                                                --SPLIT_TO_EVAL all \
                                                --TRACKER_SUB_FOLDER ""\
                                                --OUTPUT_FOLDER TrackEval/data/results \
                                                --SEQMAP_FOLDER TrackEval/data/seqmaps \
                                                --SKIP_SPLIT_FOL True \
                                                --USE_PARALLEL True \
                                                --NUM_PARALLEL_CORES 8 \
                                                --PRINT_CONFIG True \
                                                --PRINT_RESULTS False \
                                                --OUTPUT_SUMMARY True \
                                                --OUTPUT_EMPTY_CLASSES True \
                                                --OUTPUT_DETAILED True \
                                                --PLOT_CURVES True
