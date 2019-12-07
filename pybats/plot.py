import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, date
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Plot Data vs Forecast (with credible intervals)
def plot_data_forecast(fig, ax, y, f, samples, dates, linewidth=1, linecolor='b', credible_interval=95, **kwargs):
    """
    Plot observations along with sequential forecasts and credible intervals.

    :param fig: Figure to plot on (initialized from matplotlib)
    :param ax: Axes to plot on (initialized from matplotlib)
    :param y: Observations
    :param f: Point forecast, e.g. median(samples)
    :param samples: Samples from the forecast distribution (often returned from the analysis function)
    :param dates: Dates
    :param linewidth: Line width of the point forecast
    :param linecolor: Line color of the point forecast
    :param credible_interval: Size of credible interval to plot
    :param kwargs: Additional arguments to set axis style
    :return: The axes (ax)
    """

    ax.scatter(dates, y, color='k')
    ax.plot(dates, f, color=linecolor, linewidth=linewidth)
    alpha = (100 - credible_interval) / 2
    upper = np.percentile(samples, [100-alpha], axis=0).reshape(-1)
    lower = np.percentile(samples, [alpha], axis=0).reshape(-1)
    ax.fill_between(dates, upper, lower, alpha=.3, color=linecolor)

    ax = ax_style(ax, **kwargs)

    # If dates are actually dates, then format the dates on the x-axis
    if isinstance(dates[0], (datetime, date)):
        fig.autofmt_xdate()

    return ax

def plot_coef(fig, ax, coef, dates, linewidth=1, linecolor=None, legend_inside_plot=True, coef_samples=None, **kwargs):
    """
    Plot coefficients over time.

    :param fig: Figure to plot on (initialized from matplotlib)
    :param ax: Axes to plot on (initialized from matplotlib)
    :param coef: Matrix of coefficient mean values
    :param dates: Dates
    :param linewidth: Line width of the coefficient means
    :param linecolor: Line color of the coefficient means
    :param legend_inside_plot: Boolean. Put legend inside or outside of the plot?
    :param coef_samples: (optional) Samples from the distribution of the coefficient, to plot 95% credible intervals
    :param kwargs: dditional arguments to set axis style
    :return: The axes (ax)
    """

    if linecolor is not None:
        ax.plot(dates, coef, linewidth=linewidth, color=linecolor)
    else:
        ax.plot(dates, coef, linewidth=linewidth)

    # If dates are actually dates, then format the dates on the x-axis
    if isinstance(dates[0], (datetime, date)):
        fig.autofmt_xdate()

    ax = ax_style(ax, legend_inside_plot=legend_inside_plot, **kwargs)

    # Add credible intervals if samples are provided
    if coef_samples is not None:
        upper = np.percentile(coef_samples, [97.5], axis=0).reshape(-1)
        lower = np.percentile(coef_samples, [2.5], axis=0).reshape(-1)
        ax.fill_between(dates, upper, lower, alpha=.3, color=linecolor)


    # Include the y-axis labels on all subplots, which is not the matplotlib default
    ax.tick_params(labelleft=True)

    return ax

def plot_corr(fig, ax, corr, labels=None):
    """
    Plot a correlation matrix with a heatmap.

    :param fig: Figure to plot on (initialized from matplotlib)
    :param ax: Axes to plot on (initialized from matplotlib)
    :param corr: Correlation matrix
    :param labels: Labels for each entry in the correlation matrix
    :return: The axes (ax)
    """

    ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0,
                       cmap=sns.diverging_palette(10, 240, as_cmap=True),
                       cbar=True,
                       square=True, ax=ax,
                       xticklabels=labels,
                       yticklabels=labels)
    ax.set_xticklabels(labels=labels, rotation=45, size=9)
    ax.set_yticklabels(labels=labels, rotation=0, size=9)
    return ax

def ax_style(ax, ylim=None, xlim=None, xlabel=None, ylabel=None, title=None,
             legend=None, legend_inside_plot=True, topborder=False, rightborder=False, **kwargs):
    """
    A helper function to define many elements of axis style at once.

    :param ax: Axes to plot on (initialized from matplotlib)
    :param ylim: Limits for the y-axis (a list)
    :param xlim: Limits for the x-axis (a list)
    :param xlabel: Label for the x-axis
    :param ylabel: Label for the y-axis
    :param title: Title for the plot
    :param legend: Legend for the plot
    :param legend_inside_plot: Boolean. Put legend inside or outside of the plot?
    :param topborder: Boolean. Include the top border?
    :param rightborder: Boolean. Include the right border?
    :return: The axes (ax)
    """

    if legend is not None:
        if legend_inside_plot:
            ax.legend(legend)
        else:
            ax.legend(legend, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.5, frameon=False)
            # Make room for the legend
            plt.subplots_adjust(right=.85)

    if ylim is not None: ax.set_ylim(ylim)
    if xlim is not None: ax.set_xlim(xlim)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if title is not None: ax.set_title(title)

    # remove the top and right borders
    ax.spines['top'].set_visible(topborder)
    ax.spines['right'].set_visible(rightborder)

    plt.tight_layout()

    return ax