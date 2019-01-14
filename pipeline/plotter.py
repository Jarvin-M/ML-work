from os import walk
import csv
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from scipy.stats import ttest_rel, ttest_ind


class Mode(Enum):
    AUGMENTED = 1
    ORIGINAL = 2
    BOTH = 3


def read_file(file_name):
    with open(file_name, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        history = {key: [] for key in reader.fieldnames}
        for row in reader:
            for key in reader.fieldnames:
                history[key].append(float(row[key]))
        return history


def average_history(histories):
    avg_history = {}
    for history in histories:
        if avg_history == {}:
            avg_history = {key: np.array(history[key]) for key in history.keys()}
        else:
            for key in history.keys():
                avg_history[key] = np.array(history[key]) + avg_history[key]
    for key in avg_history.keys():
        avg_history[key] /= len(histories)
    return avg_history


def get_histories(folder_name):
    augmented_filenames = []
    original_filenames = []
    for (dirpath, dirnames, filenames) in walk(folder_name):
        if dirpath[-9:] == 'augmented':
            augmented_filenames.extend([dirpath + '/' + name for name in filenames if name[:4] == 'out_'])
        if dirpath[-8:] == 'original':
            original_filenames.extend([dirpath + '/' + name for name in filenames if name[:4] == 'out_'])
    augmented_filenames.sort()
    original_filenames.sort()
    augmented_histories = [read_file(file_name) for file_name in augmented_filenames]
    original_histories = [read_file(file_name) for file_name in original_filenames]

    return augmented_histories, original_histories


def average_folder(folder_name):
    augmented_histories, original_histories = get_histories(folder_name)
    return average_history(augmented_histories), average_history(original_histories)


def print_folder(folder_name, key):
    augmented_histories, original_histories = get_histories(folder_name)
    mode = 1
    if mode == 1:  # Best performance for each run
        augmented_maxes = np.array([max(history[key]) for history in augmented_histories])*100
        original_maxes = np.array([max(history[key]) for history in original_histories])*100
    elif mode == 2:  # Last performance for each run
        augmented_maxes = np.array([history[key][-1] for history in augmented_histories])*100
        original_maxes = np.array([history[key][-1] for history in original_histories])*100
    else:  # average last 50 performances for each run
        augmented_maxes = np.array([sum(history[key][-50:])/50 for history in augmented_histories])*100
        original_maxes = np.array([sum(history[key][-50:])/50 for history in original_histories])*100

    folder_string = '_'.join(folder_name.split('_')[:2]) + '_' + '_'.join(folder_name.split('_')[-2:])
    differences = augmented_maxes-original_maxes[:augmented_maxes.shape[0]]
    two_sided_pvalue = ttest_rel(augmented_maxes, original_maxes[:augmented_maxes.shape[0]]).pvalue
    print("{}\t\t{:.2f}±{:.2f}\t{:.2f}±{:.2f}\t{:.2f}±{:.2f}\t{:.5f}".format(folder_string,
                                                                             np.mean(augmented_maxes),
                                                                             np.std(augmented_maxes),
                                                                             np.mean(original_maxes),
                                                                             np.std(original_maxes),
                                                                             np.mean(differences),
                                                                             np.std(differences),
                                                                             two_sided_pvalue
                                                                             ))


def plot_folders(folder_names, key, y_label='accuracy', zoom=False, mode=Mode.BOTH):
    legend = []
    epochs = 0
    colors = ['r', 'g', 'b', 'c', 'y', 'm', 'k']
    print('\n\t\t\t\t\taugmented\toriginal\tdifference\tp-value')
    for folder_name, color in zip(folder_names, colors):
        print_folder(folder_name, key)
        augmented_hist, original_hist = average_folder(folder_name)
        experiment_name = folder_name  # ' '.join(folder_name.split('_')[-2:])
        if mode == Mode.AUGMENTED or mode == Mode.BOTH and augmented_hist:
            plt.plot(augmented_hist[key], color)
            legend.append(experiment_name+" augmented")
        if mode == Mode.ORIGINAL or mode == Mode.BOTH and original_hist:
            plt.plot(original_hist[key], color+'--')
            legend.append(experiment_name+" original")
        epochs = len(augmented_hist[key] if augmented_hist else original_hist[key])

    plt.axis(xmin=epochs*3/4 if zoom else 0, xmax=epochs-1, ymin=0.8 if zoom else 0, ymax=1.)
    plt.title(key)
    plt.ylabel(y_label)
    plt.xlabel('epoch')
    plt.legend(legend, loc='lower right')
    #plt.show()
    plt.close()

# 26_12_2018 overview
# plot_folders(['26_12_2018_split_01', '26_12_2018_split_02', '26_12_2018_split_05', '26_12_2018_split_08'], 'val_acc', zoom=True)
# plot_folders(['26_12_2018_split_01', '26_12_2018_split_02'], 'val_acc', zoom=True)

# 27_12_2018 overview
# plot_folders(['27_12_2018_split_01', '27_12_2018_split_02', '27_12_2018_split_005'], 'val_acc', zoom=True)
# plot_folders(['27_12_2018_split_01', '27_12_2018_split_02', '26_12_2018_split_01', '26_12_2018_split_02'], 'val_acc', zoom=True)

# 28_12_2018 overview
# plot_folders(['28_12_2018_split_01', '28_12_2018_split_02', '28_12_2018_split_005', '28_12_2018_split_08'], 'val_acc', zoom=True)
# 27-28 comparison
# plot_folders(['28_12_2018_split_01', '28_12_2018_split_02', '28_12_2018_split_005', '27_12_2018_split_01', '27_12_2018_split_02', '27_12_2018_split_005'], 'val_acc', zoom=True, mode=Mode.ORIGINAL)
# plot_folders(['28_12_2018_split_01', '28_12_2018_split_02', '28_12_2018_split_005', '27_12_2018_split_01', '27_12_2018_split_02', '27_12_2018_split_005'], 'val_acc', zoom=True, mode=Mode.AUGMENTED)

# 29_12_2018 overview
# plot_folders(['29_12_2018_split_01', '29_12_2018_split_02', '29_12_2018_split_005', '29_12_2018_split_08'], 'val_acc', zoom=False)
# 28-29 comparison 0.05 0.1 0.2
# plot_folders(['28_12_2018_split_01', '28_12_2018_split_02', '28_12_2018_split_005', '29_12_2018_split_01', '29_12_2018_split_02', '29_12_2018_split_005'], 'val_acc', zoom=True, mode=Mode.ORIGINAL)
# plot_folders(['28_12_2018_split_01', '28_12_2018_split_02', '28_12_2018_split_005', '29_12_2018_split_01', '29_12_2018_split_02', '29_12_2018_split_005'], 'val_acc', zoom=True, mode=Mode.AUGMENTED)
# 28-29 comparison 0.2 0.8
# plot_folders(['28_12_2018_split_08', '28_12_2018_split_02', '29_12_2018_split_08', '29_12_2018_split_02'], 'val_acc', zoom=True, mode=Mode.ORIGINAL)
# plot_folders(['28_12_2018_split_08', '28_12_2018_split_02', '29_12_2018_split_08', '29_12_2018_split_02'], 'val_acc', zoom=True, mode=Mode.AUGMENTED)

# 01_01_2019 overview
# plot_folders(['01_01_2019_split_01', '01_01_2019_split_02', '01_01_2019_split_005', '01_01_2019_split_08'], 'val_acc', zoom=False)

# 09_01_2019 overview
# plot_folders(['09_01_2019_split_01', '09_01_2019_split_02', '09_01_2019_split_005', '09_01_2019_split_08'], 'val_acc', zoom=False)

# 10_01_2019 overview
# plot_folders(['10_01_2019_split_01', '10_01_2019_split_02', '10_01_2019_split_005', '10_01_2019_split_08'], 'val_acc', zoom=True)

# 12_01_2019 overview
plot_folders(['12_01_2019_split_01', '12_01_2019_split_02', '12_01_2019_split_005', '12_01_2019_split_08'], 'val_acc', zoom=True)

# 13_01_2019 overview
# plot_folders(['13_01_2019_split_01', '13_01_2019_split_02', '13_01_2019_split_005', '13_01_2019_split_08'], 'val_acc', zoom=True)

# 12-13 comparison 0.05 0.1
# plot_folders(['12_01_2019_split_01', '12_01_2019_split_005', '13_01_2019_split_01', '13_01_2019_split_005'], 'val_acc', zoom=False, mode=Mode.ORIGINAL)
# plot_folders(['12_01_2019_split_01', '12_01_2019_split_005', '13_01_2019_split_01', '13_01_2019_split_005'], 'val_acc', zoom=False, mode=Mode.AUGMENTED)
# 12-13 comparison 0.2 0.8
# plot_folders(['12_01_2019_split_02', '12_01_2019_split_08', '13_01_2019_split_02', '13_01_2019_split_08'], 'val_acc', zoom=False, mode=Mode.ORIGINAL)
# plot_folders(['12_01_2019_split_02', '12_01_2019_split_08', '13_01_2019_split_02', '13_01_2019_split_08'], 'val_acc', zoom=False, mode=Mode.AUGMENTED)

# 14_01_2019 overview
plot_folders(['14_01_2019_split_01', '14_01_2019_split_02', '14_01_2019_split_005', '14_01_2019_split_08'], 'val_acc', zoom=True)
