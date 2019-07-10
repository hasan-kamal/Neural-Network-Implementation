import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def show_confusion_matrix(y_predicted, y_true, label_to_string_map=None):
    """
    Plots confusion matrix between provided labels.
    
    Parameters:
        y_predicted (np array)    : predicted labels, dim ( num samples, )
        y_true      (np array)    : true labels,      dim ( num samples, )
        label_to_string_map (dict): map specifying string to be displayed for each label
    """
    
    labels, counts = np.unique(y_true, return_counts=True)
    map_label_to_index = { label : i for i, label in enumerate(labels) }
    
    # compute confusion matrix
    confusion_matrix = np.zeros((labels.shape[0], labels.shape[0]))
    for prediction_label, true_label in zip(y_predicted, y_true):
        confusion_matrix[map_label_to_index[true_label], map_label_to_index[prediction_label]] += 1
    
    # plot confusion matrix
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    fig.suptitle('Confusion matrix', fontsize=14, y=0.9)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    labels_yaxes = [ '{} [{}]'.format(label_to_string_map[label], label) for i, label in enumerate(labels) ]
    labels_xaxes = [ '[{}] {}'.format(label, label_to_string_map[label]) for i, label in enumerate(labels) ]
    ax.set_xticklabels([''] + labels_xaxes)
    ax.set_yticklabels([''] + labels_yaxes)
    im = ax.matshow(confusion_matrix, cmap='coolwarm')
    # attribution for following two lines: https://stackoverflow.com/questions/20998083/show-the-values-in-the-grid-using-matplotlib
    for (i, j), z in np.ndenumerate(confusion_matrix):
        ax.text(j, i, '{}'.format(z), ha='center', va='center')
    fig.colorbar(im)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.tick_params(axis='x', labelrotation=60)
    plt.show()