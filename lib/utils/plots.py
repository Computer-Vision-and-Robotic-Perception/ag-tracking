import numpy as np
import matplotlib.pyplot as plt

metrics = {
'Tracktor': {
    'MOTA_values': [93.006, 93.051, 95.622, 94.777, 93.825, 92.863, 93.811, 94.597],
    'HOTA_values': [69.550, 69.779, 68.985, 70.660, 94.299, 91.845, 93.815, 94.768],
    'DetA_values': [99.627, 99.321, 99.145, 98.967, 99.538, 99.438, 99.435, 99.164],
    'IDF1_values': [57.171, 52.513, 55.913, 55.950, 93.496, 90.155, 93.729, 94.570],},

'Tracktor++': {
    'MOTA_values': [96.855, 96.893, 98.249, 97.831, 99.009, 98.125, 97.937, 98.535],
    'HOTA_values': [52.306, 43.609, 65.203, 59.103, 58.887, 49.548, 50.227, 55.688],
    'DetA_values': [99.583, 99.355, 99.114, 99.009, 99.538, 99.438, 99.435, 99.212],
    'IDF1_values': [43.719, 29.000, 58.368, 49.651, 36.687, 26.757, 27.433, 33.010],},

'PlantTracktor': {
    'MOTA_values': [94.722, 95.271, 94.793, 93.365, 96.249, 95.867, 96.036, 95.989],
    'HOTA_values': [72.775, 71.313, 71.298, 70.829, 96.273, 95.891, 96.092, 96.037],
    'DetA_values': [95.676, 96.525, 95.310, 94.281, 96.253, 95.874, 96.045, 96.028],
    'IDF1_values': [60.415, 60.577, 60.008, 60.352, 95.069, 95.893, 95.983, 95.949],}, # 60, 54, 60, 57, 98, 97, 97, 97

'PlantTracktor+': {
    'MOTA_values': [98.120, 98.078, 97.341, 95.377, 98.536, 98.173, 98.173, 98.087],
    'HOTA_values': [98.138, 98.940, 74.112, 72.608, 98.524, 98.199, 98.199, 97.104],
    'DetA_values': [99.123, 99.117, 98.888, 96.363, 99.559, 99.357, 99.357, 99.165],
    'IDF1_values': [98.560, 98.420, 61.295, 58.055, 98.716, 98.496, 98.496, 98.509],},
}


def plot_metric(t, data, metric, title, xlabel, ylabel, fontsize=20): # data includes ordenated, style and label
    for dat in data:
        plt.plot(t, dat[0], dat[1], label=dat[2])
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize*0.8)
    plt.xticks(fontsize=fontsize*0.8)
    plt.yticks(fontsize=fontsize*0.8)
    plt.ylim(-5, 105)
    plt.legend(fontsize=fontsize*0.8)
    plt.title(title, fontsize=fontsize)
    plt.grid()


if __name__ == "__main__":
    # First part
    for k, v in metrics.items():
        for metric, values in v.items():
            vals = np.array(values)
            print(f'{k} {metric} {vals.mean():.3f} +- {vals.std():.3f}')

    # Second part
    t =        np.array([ 0.0,    0.2,    0.4,    0.5,    0.6,    0.8,    1.0])

    hota_nms = np.array([87.719, 91.978, 90.272, 90.778, 91.322, 87.463, 21.741])
    mota_nms = np.array([97.845, 97.736, 97.318, 97.459, 97.655, 97.611,  0.000])
    idf1_nms = np.array([86.530, 88.818, 84.410, 86.579, 86.877, 85.204, 19.677])

    hota_dth = np.array([66.048, 88.406, 89.509, 91.978, 87.859, 86.135, 0])
    mota_dth = np.array([81.172, 97.305, 97.434, 97.736, 97.298, 97.374, 0])
    idf1_dth = np.array([73.149, 85.777, 86.403, 88.818, 86.390, 83.888, 0])

    hota_rth = np.array([67.546, 87.362, 89.234, 91.978, 88.508, 87.047, 0])
    mota_rth = np.array([81.545, 97.414, 97.051, 97.736, 96.017, 95.607, 0])
    idf1_rth = np.array([74.297, 86.454, 86.161, 88.818, 86.097, 85.155, 0])

    # By metric
    plt.figure(figsize=(13, 4))

    plt.subplot(1, 3, 1)
    plot_metric(t, [[mota_nms, 's-', '$\lambda_{nms}$'], 
                    [mota_dth, '*-', '$s_{new}$'], 
                    [mota_rth, '.-', '$s_{active}$']], 
                   'MOTA', '', 'Parameter', 'MOTA')
    plt.subplot(1, 3, 2)
    plot_metric(t, [[hota_nms, 's-', '$\lambda_{nms}$'],
                    [hota_dth, '*-', '$s_{new}$'],
                    [hota_rth, '.-', '$s_{active}$']],
                   'HOTA', '', 'Parameter', 'HOTA')
    plt.subplot(1, 3, 3)
    plot_metric(t, [[idf1_nms, 's-', '$\lambda_{nms}$'],
                    [idf1_dth, '*-', '$s_{new}$'],
                    [idf1_rth, '.-', '$s_{active}$']],
                   'IDF1', '', 'Parameter', 'IDF1')

    plt.tight_layout()
    plt.savefig('output/LettuceMOT/parameters/parameter_by_metric.jpg')
    plt.savefig('output/LettuceMOT/parameters/parameter_by_metric.eps')

    # By parameter
    plt.figure(figsize=(13, 4))

    plt.subplot(1, 3, 1)
    plot_metric(t, [[mota_nms, 's-', 'MOTA'],
                    [hota_nms, '*-', 'HOTA'],
                    [idf1_nms, '.-', 'IDF1']],
                   'NMS Threshold', '', '$\lambda_{nms}$', 'Metric')
    plt.subplot(1, 3, 2)
    plot_metric(t, [[mota_dth, 's-', 'MOTA'],
                    [hota_dth, '*-', 'HOTA'],
                    [idf1_dth, '.-', 'IDF1']],
                   'Detection Threshold', '', '$s_{new}$', 'Metric')
    plt.subplot(1, 3, 3)
    plot_metric(t, [[mota_rth, 's-', 'MOTA'],
                    [hota_rth, '*-', 'HOTA'],
                    [idf1_rth, '.-', 'IDF1']],
                   'Regression Threshold', '', '$s_{active}$', 'Metric')
    plt.tight_layout()
    plt.savefig('output/LettuceMOT/parameters/metric_by_parameter.jpg')
    plt.savefig('output/LettuceMOT/parameters/metric_by_parameter.eps')

    # Without grouping
    plt.figure(figsize=(13, 10))

    plt.subplot(3, 3, 1)
    plot_metric(t, [[mota_nms, 's-b', 'MOTA']], 'MOTA', 'MOTA vs. NMS Threshold', '$\lambda_{nms}$', '')
    plt.subplot(3, 3, 2)
    plot_metric(t, [[mota_dth, '*-g', 'MOTA']], 'MOTA', 'MOTA vs. Detection Threshold', '$s_{new}$', '')
    plt.subplot(3, 3, 3)
    plot_metric(t, [[mota_rth, '.-r', 'MOTA']], 'MOTA', 'MOTA vs. Regression Threshold', '$s_{active}$', '')
    plt.subplot(3, 3, 4)
    plot_metric(t, [[hota_nms, 's-b', 'HOTA']], 'HOTA', 'HOTA vs. NMS Threshold', '$\lambda_{nms}$', '')
    plt.subplot(3, 3, 5)
    plot_metric(t, [[hota_dth, '*-g', 'HOTA']], 'HOTA', 'HOTA vs. Detection Threshold', '$s_{new}$', '')
    plt.subplot(3, 3, 6)
    plot_metric(t, [[hota_rth, '.-r', 'HOTA']], 'HOTA', 'HOTA vs. Regression Threshold', '$s_{active}$', '')
    plt.subplot(3, 3, 7)
    plot_metric(t, [[idf1_nms, 's-b', 'IDF1']], 'IDF1', 'IDF1 vs. NMS Threshold', '$\lambda_{nms}$', '')
    plt.subplot(3, 3, 8)
    plot_metric(t, [[idf1_dth, '*-g', 'IDF1']], 'IDF1', 'IDF1 vs. Detection Threshold', '$s_{new}$', '')
    plt.subplot(3, 3, 9)
    plot_metric(t, [[idf1_rth, '.-r', 'IDF1']], 'IDF1', 'IDF1 vs. Regression Threshold', '$s_{active}$', '')
    plt.tight_layout()
    plt.savefig('output/LettuceMOT/parameters/all.jpg')
    plt.savefig('output/LettuceMOT/parameters/all.eps')
    plt.show()

    



