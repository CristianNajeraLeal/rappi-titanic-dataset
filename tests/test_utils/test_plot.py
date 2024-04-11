import os
import numpy as np
from src.utils.plot import create_confusion_matrix_plot


def test_create_confusion_matrix_plot():
    # Prepare a small dataset to test
    y_test_set = np.array([0, 1, 0, 1])
    y_predicted = np.array([0, 1, 0, 0])
    plot_file = 'test_confusion_matrix.png'

    # Ensure the plot file does not exist before testing
    if os.path.exists(plot_file):
        os.remove(plot_file)

    # Call the function under test
    create_confusion_matrix_plot(y_test_set, y_predicted, plot_file)

    # Check if the plot file was created
    assert os.path.exists(plot_file), "Plot file should exist after calling create_confusion_matrix_plot"

    # Optionally, you could also check the file size to ensure it's not empty
    assert os.path.getsize(plot_file) > 0, "Plot file should not be empty"

    # Cleanup: Remove the plot file after the test to avoid clutter
    os.remove(plot_file)
