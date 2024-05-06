import os
import numpy as np
import matplotlib.pyplot as plt

def create_histograms_from_path(p, bins, range):
    files = os.listdir(p)
    all = np.array([])
    for file in files:
        arr = np.loadtxt(p + file)
        all = np.concatenate([all, arr])
        plt.hist(arr, bins, range)
        plt.savefig(p + file[:-4] + '.jpg')
        plt.clf()
    plt.hist(all, bins, range)
    plt.savefig(p + 'all.jpg')
    plt.clf()
    plt.plot(all)
    plt.savefig(p + 'plot.jpg')

def obtain_summary_dict(p):
    file = open(p + '/pedestrian_summary.txt')
    lines = file.readlines()
    line1, line2 = lines[0], lines[1]
    line1 = line1[:-1].split(' ')
    line2 = line2[:-1].split(' ')
    summary = {}
    for i, met in enumerate(line1):
        summary[met] = float(line2[i])
    return summary

def create_summary_from_csv(p):
    # read file
    file = open(p + '/pedestrian_detailed.csv')
    # metrics to extract
    metrics = ['HOTA', 'DetA', 'AssA', 'AssRe', 'AssPr']
    countable_metrics = ['MOTA', 'MOTP', 'IDSW', 'IDF1', 'IDR', 'IDP'] # not dependent of alpha
    extract_perc = 50                                                  # alpha/100
    extract_name = [met + '___%d' % extract_perc for met in metrics]   # names
    # add countable ones:
    metrics += countable_metrics
    extract_name += countable_metrics
    # initialize sequences and values
    seqs = []
    values = []
    for i, line in enumerate(file.readlines()):
        fields = line[:-1].split(',')
        if i:
            seqs.append(fields[0])
            values.append([float(val)*100 for val in fields[1:]])
        else:
            names = fields[1:]
    values = np.array(values)
    # Separate metrics
    all = {}
    for i, seq in enumerate(seqs):
        all[seq] = {}
        for j, met in enumerate(extract_name):
            all[seq][metrics[j]] = values[i, names.index(met)]
            print(seq, '|', metrics[j] ,'= %3.3f' % values[i, names.index(met)])
        pr = ''
        for met in ['MOTA', 'HOTA', 'DetA', 'AssA', 'AssRe', 'AssPr', 'IDF1', 'IDR', 'IDP', 'IDSW']:
            pr += ' & %3.3f' % all[seq][met]  
        print(pr)
        print('-------------------------------------')
    return all

if __name__ == '__main__':
    pass
    # path = os.getcwd() + "/output/reids/"
    # create_histograms_from_path(path, 25, (0, 2.5))

    path = 'TrackEval/data/results_lettuce/exp06'
    create_summary_from_csv(path)