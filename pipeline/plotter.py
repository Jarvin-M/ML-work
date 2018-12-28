from os import walk
import csv
import numpy as np
import matplotlib.pyplot as plt


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


def read_folder(folder_name):
    augmented_filenames = []
    original_filenames = []
    for (dirpath, dirnames, filenames) in walk(folder_name):
        if dirpath[-9:] == 'augmented':
            augmented_filenames.extend([dirpath + '/' + name for name in filenames if name[:4] == 'out_'])
        if dirpath[-8:] == 'original':
            original_filenames.extend([dirpath + '/' + name for name in filenames if name[:4] == 'out_'])
    augmented_histories = [read_file(file_name) for file_name in augmented_filenames]
    original_histories = [read_file(file_name) for file_name in original_filenames]

    return average_history(augmented_histories), average_history(original_histories)


def plot_folders(folder_names, key, y_label='accuracy', zoom=False):
    legend = []
    epochs = 0
    colors = ['r', 'g', 'b', 'c', 'y']
    for folder_name, color in zip(folder_names, colors):
        augmented_hist, original_hist = read_folder(folder_name)
        plt.plot(augmented_hist[key], color)
        plt.plot(original_hist[key], color+'--')
        experiment_name = ' '.join(folder_name.split('_')[-2:])
        legend.extend([experiment_name+" augmented", experiment_name+" original"])
        epochs = len(original_hist[key])

    plt.axis(xmin=epochs*3/4 if zoom else 0, xmax=epochs-1, ymin=0.8 if zoom else 0, ymax=1.)
    plt.title(key)
    plt.ylabel(y_label)
    plt.xlabel('epoch')
    plt.legend(legend, loc='lower right')
    plt.show()
    plt.close()


# plot_folders(['26_12_2018_split_01', '26_12_2018_split_02', '26_12_2018_split_05', '26_12_2018_split_08'], 'val_acc', zoom=True)
# plot_folders(['26_12_2018_split_01', '26_12_2018_split_02'], 'val_acc', zoom=True)
# plot_folders(['27_12_2018_split_01', '27_12_2018_split_02', '27_12_2018_split_005'], 'val_acc', zoom=True)
plot_folders(['27_12_2018_split_01', '27_12_2018_split_02', '26_12_2018_split_01', '26_12_2018_split_02'], 'val_acc', zoom=True)
