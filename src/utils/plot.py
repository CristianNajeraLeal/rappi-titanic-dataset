"""
This module provides functionalities to create png file for confusion matrix plots.

Functions:
    create_confusion_matrix_plot(y_test_set, y_predicted, plot_file): Generates a confusion
        matrix plot and saves it to a specified file.
"""


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def create_confusion_matrix_plot(y_test_set, y_predicted, plot_file):
    """
    Generate and save a confusion matrix plot comparing test and predicted values.
    :param y_test_set: The true labels from the test dataset.
    :param y_predicted: The labels predicted by a model.
    :param plot_file: The file path where the plot should be saved.
    :return: None
    """
    cm = confusion_matrix(y_test_set, y_predicted)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues", cbar=False)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])

    fig.savefig(plot_file)
    plt.close(fig)
