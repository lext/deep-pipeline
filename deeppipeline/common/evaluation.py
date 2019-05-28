import numpy as np
import matplotlib.pyplot as plt


def cumulative_error_plot(errors, labels, title, colors=None, units='mm'):
    """

    Parameters
    ----------
    errors : Array of errors
        Errors for the whole dataset for each landmark. Each sample in the dataset is represented by a row.
        The error for each landmark is stored in columns.
    labels : tuple or list of str or None
        Labels for each landmark
    title : str or None
        Title for the plot
    colors : list of str or None
        Colors for each landmark
    units : str
        Units to be displayed on X-axis

    Returns
    -------
    out : None
        Plots the cumulative curves.
    """
    plt.figure(figsize=(5, 5))
    for i in range(errors.shape[1]):
        sorted_data = np.sort(errors[:, i])
        if labels is not None:
            if colors is not None:
                plt.step(sorted_data, np.arange(sorted_data.size) / sorted_data.size, label=labels[i], colors=colors[i])
            else:
                plt.step(sorted_data, np.arange(sorted_data.size) / sorted_data.size, label=labels[i])
        else:
            if colors is not None:
                plt.step(sorted_data, np.arange(sorted_data.size) / sorted_data.size, color=colors[i])
            else:
                plt.step(sorted_data, np.arange(sorted_data.size) / sorted_data.size)

    plt.xlim(0, 4)
    plt.ylim(0, 1)
    plt.ylabel('Recall [%]')
    plt.xlabel(f'Distance from GT [f{units}]')
    plt.grid()
    if title is not None:
        plt.title(title)
    if labels is not None:
        plt.legend()
    plt.show()
