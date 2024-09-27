import os
import sys
import numpy as np
from tabulate import tabulate
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval

def populate_seq_track(dict_by_tracker, dict_by_seq, tracker, seq, metricname, metricvalue):
    if tracker not in dict_by_tracker:
        dict_by_tracker[tracker] = {}
    if seq not in dict_by_tracker[tracker]:
        dict_by_tracker[tracker][seq] = {}
    if metricname not in dict_by_tracker[tracker][seq]:
        dict_by_tracker[tracker][seq][metricname] = metricvalue

    if seq not in dict_by_seq:
        dict_by_seq[seq] = {}
    if tracker not in dict_by_seq[seq]:
        dict_by_seq[seq][tracker] = {}
    if metricname not in dict_by_seq[seq][tracker]:
        dict_by_seq[seq][tracker][metricname] = metricvalue

    return dict_by_tracker, dict_by_seq

def sequence_data_table(res):
    table_by_tracker = {}
    table_by_sequence = {}
    
    for tracker, trackervalues in res[0]['MotChallenge2DBox'].items():
        for seq, seqvalues in trackervalues.items():
            for _, classvalues in seqvalues.items(): # obj-class
                for _, metrictypevalues in classvalues.items():
                    for metricname, metricvalue in metrictypevalues.items():
                        value = np.mean(metricvalue)
                        table_by_tracker, table_by_sequence = \
                            populate_seq_track( table_by_tracker, 
                                                table_by_sequence, 
                                                tracker, 
                                                seq, 
                                                metricname, 
                                                value)
    return table_by_sequence, table_by_tracker

def render_table(table_results, metrics, sequences, trackers, group_by="trackers"):
    table_data = []
    if group_by == "trackers":
        groups = [trackers, sequences]
        headers = ["Tracker", "Sequences"] + metrics

    else: # group_by == "sequences":
        groups = [sequences, trackers]
        headers = ["Sequence", "Tracker"] + metrics
    
    pre_el1 = None
    for element1 in groups[0]:
        for element2 in groups[1]:
            if element1 == pre_el1: 
                col = ["", element2]
            else:                   
                col = [element1, element2]
                if pre_el1 is not None:
                    table_data.append(["-----"] * len(headers))
            pre_el1 = element1
            for metric in metrics:
                col.append('%.2f' % (table_results.get(element1, {}).get(element2, {}).get(metric, '') * 100))
            table_data.append(col)

    print('\n\n' + tabulate(table_data, headers=headers) + "\n")

def analyze_custom_track_eval(params, metrics, sequences, trackers):
    # Get default configurations
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    # Filter configurations
    eval_config = {k: v for k, v in params.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in params.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in params.items() if k in default_metrics_config.keys()}
    # Create evaluator
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    # Evaluate
    res = evaluator.evaluate(dataset_list, metrics_list)
    # Collect results
    dict_by_sequence, dict_by_tracker = sequence_data_table(res)
    # Render tables
    render_table(dict_by_sequence, metrics, sequences, trackers, 'sequences')
    render_table(dict_by_tracker, metrics, sequences, trackers, 'trackers')

def track_eval(cfg, benchmark):
    freeze_support()
    params = {
        "GT_FOLDER": cfg['datadir'],
        "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
        "SEQMAP_FOLDER": ["data/seqmaps"],
        "SEQMAP_FILE": None,
        "SEQ_INFO": None,
        "TRACKERS_FOLDER": cfg['outdir'] + 'trackers', 
        "OUTPUT_FOLDER": cfg['outdir'] + 'trackers-results',
        "OUTPUT_SUB_FOLDER": "",
        "LOG_ON_ERROR": cfg['outdir'] + 'trackers-results/error.log',
        "BENCHMARK": benchmark,
        "SPLIT_TO_EVAL": "all",
        "CLASSES_TO_EVAL": ["pedestrian"],
        "TRACKERS_TO_EVAL": None,
        "TRACKER_SUB_FOLDER": "",
        "TRACKER_DISPLAY_NAMES": None,
        "METRICS": ["HOTA", "CLEAR", "Identity"], 
        "THRESHOLD": 0.5,
        "USE_PARALLEL": True,
        "NUM_PARALLEL_CORES": 16,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_DETAILED": True,
        "OUTPUT_EMPTY_CLASSES": True,
        "PRINT_CONFIG": False,
        "PRINT_RESULTS": False,
        "PRINT_ONLY_COMBINED": False,
        "DISPLAY_LESS_PROGRESS": True, 
        "PLOT_CURVES": True,
        "TIME_PROGRESS": True,
        "INPUT_AS_ZIP": False,
        "DO_PREPROC": True,
        "SKIP_SPLIT_FOL": True,
        "BREAK_ON_ERROR": True,
        "RETURN_ON_ERROR": False,
    }
    metrics = cfg['track_metrics_eval']
    sequences = cfg['sets'] + ["COMBINED_SEQ"]
    trackers = sorted([tracker for tracker in os.listdir(cfg['outdir'] + 'trackers')])
    
    print("Saving output to:", cfg['outdir'] + 'track_eval_%s.log' % benchmark)
    sys.stdout = open(cfg['outdir'] + 'track_eval_%s.log' % benchmark, 'w')
    analyze_custom_track_eval(params, metrics, sequences, trackers)
    sys.stdout.close() 


if __name__ == "__main__":
    freeze_support()
    params = {
        "GT_FOLDER": "data/AppleMOTS/MOT",
        "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
        "SEQMAP_FOLDER": ["data/seqmaps"],
        "SEQMAP_FILE": None,
        "SEQ_INFO": None,
        "TRACKERS_FOLDER": "output/AppleMOTS/MOT/trackers", 
        "OUTPUT_FOLDER": ["output/AppleMOTS/MOT/trackers-results"],
        "OUTPUT_SUB_FOLDER": "",
        "LOG_ON_ERROR": "output/AppleMOTS/MOT/trackers-results/error.log",
        "BENCHMARK": "AppleMOT",
        "SPLIT_TO_EVAL": "all",
        "CLASSES_TO_EVAL": ["pedestrian"],
        "TRACKERS_TO_EVAL": None,
        "TRACKER_SUB_FOLDER": "",
        "TRACKER_DISPLAY_NAMES": None,
        "METRICS": ["HOTA", "CLEAR", "Identity"], 
        "THRESHOLD": 0.5,
        "USE_PARALLEL": True,
        "NUM_PARALLEL_CORES": 16,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_DETAILED": True,
        "OUTPUT_EMPTY_CLASSES": True,
        "PRINT_CONFIG": False,
        "PRINT_RESULTS": False,
        "PRINT_ONLY_COMBINED": False,
        "DISPLAY_LESS_PROGRESS": True, 
        "PLOT_CURVES": True,
        "TIME_PROGRESS": True,
        "INPUT_AS_ZIP": False,
        "DO_PREPROC": True,
        "SKIP_SPLIT_FOL": True,
        "BREAK_ON_ERROR": True,
        "RETURN_ON_ERROR": False,
    }
    metrics = ["MOTA", "HOTA", "DetA", "AssA", "AssRe", "AssPr", "IDF1", "IDR", "IDP"]
    sequences = ["0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0010", "0011", "0012", "COMBINED_SEQ"]
    trackers = ["ag", "bytetrack", "clean"]
    analyze_custom_track_eval(params, metrics, sequences, trackers)
