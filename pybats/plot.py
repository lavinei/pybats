import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Plot Data vs Forecast (with credible intervals)
def plot_data_forecast(fig, ax, y, f, samples, dates, linewidth=1, linecolor='b', credible_interval=95, **kwargs):

    ax.scatter(dates, y, color='k')
    ax.plot(dates, f, color=linecolor, linewidth=linewidth)
    alpha = (100 - credible_interval) / 2
    upper = np.percentile(samples, [100-alpha], axis=0).reshape(-1)
    lower = np.percentile(samples, [alpha], axis=0).reshape(-1)
    ax.fill_between(dates, upper, lower, alpha=.3, color=linecolor)

    ax = ax_style(ax, **kwargs)

    # If dates are actually dates, then format the dates on the x-axis
    if isinstance(dates[0], datetime):
        fig.autofmt_xdate()

    return ax

def plot_coef(fig, ax, coef, dates, linewidth=1, linecolor=None, legend_inside_plot=True, coef_samples=None, **kwargs):


        if linecolor is not None:
            ax.plot(dates, coef, linewidth=linewidth, color=linecolor)
        else:
            ax.plot(dates, coef, linewidth=linewidth)

        # If dates are actually dates, then format the dates on the x-axis
        if isinstance(dates[0], datetime):
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
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    return ax