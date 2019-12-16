import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config import IowaHousePricingConfig as cfg
from utils import extract_data


def draw_boxplot(categorical_x: str, numerical_y: str, filename: str, data: pd.DataFrame):
    """
    Draw a box plot applied for catergorical variable in x-axis vs numerical variable in y-axis
    :param categorical_x: categorical x variable
    :param numerical_y:  numerical y variable
    :param filename: name of file to save the fig to
    :param data: data frame
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    sns.boxplot(x=categorical_x, y=numerical_y, data=data, ax=ax)
    if categorical_x in ['Neighborhood', 'Exterior1st', 'Exterior2nd']:
            plt.sca(ax)
            plt.xticks(rotation=70) # too many variables on X axis so we need to rotate the names to that they wont interfere
    ax.set_title(filename)
    fig.tight_layout()
    fig.savefig(cfg.visualization_dir + "\{}.png".format(filename))


def draw_histograms(df, variables, n_rows, n_cols, filename="histograms", save_fig=True, show_fig=True, fig_size=(50,50)):
    """
    drawing histograms on demand, saves histogram in visualization dir
    :param df: the data frame from where to draw the initial histograms
    :param variables: what variable from the df we are interested in
    :param n_rows: number of rows in fig
    :param n_cols: number of columns in fig
    """
    if fig_size is not None:
        fig = plt.figure(figsize=fig_size)
    else:
        fig = plt.figure()
    for i, var_name in enumerate(variables):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        df[var_name].hist(ax=ax)
        plt.axvline(df[var_name].mean(), color='k', linestyle='dashed', linewidth=2, label='Mean')
        plt.axvline(df[var_name].median(), color='r', linestyle='dashed', linewidth=2, label='Median')
        ax.set_title(var_name+" Distribution")
        plt.legend()
    fig.tight_layout()
    if save_fig:
        fig.savefig(cfg.visualization_dir + "\{}.png".format(filename))
    if show_fig:
        plt.show()


def draw_scatters(df, variables, n_rows, n_cols, filename="scatter_plot", y='SalePrice', y_label='Sale Price',
                  save_fig=True, show_fig=True):
    """
    drawing scatter plots on demand, saves scatter plots in visualization dir
    :param df: the data frame from where to draw the initial scatter_plot
    :param variables: what variable from the df we are interested in
    :param n_rows: number of rows in fig
    :param n_cols: number of columns in fig
    """
    # fig = plt.figure(figsize=(50, 50))
    fig = plt.figure()
    for i, var_name in enumerate(variables):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        sns.regplot(x=var_name, y=y, data=df, fit_reg=False, scatter_kws={'alpha': 0.2})
        ax.set_title(var_name + " vs. {}".format(y_label))
    fig.tight_layout()

    if save_fig:
        plt.savefig(cfg.visualization_dir + "\{}.png".format(filename))
    if show_fig:
        plt.show()


def draw_bars(df, variables, n_rows, n_cols, filename="bar_plot"):
    """
    drawing bar plots on demand, saves bar plots in visualization dir
    :param df: the data frame from where to draw the initial bar_plot
    :param variables: what variable from the df we are interested in
    :param n_rows: number of rows in fig
    :param n_cols: number of columns in fig
    """
    fig=plt.figure(figsize=(40, 110))
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows, n_cols, i+1)
        sns.barplot(x=var_name, y='SalePrice', data=df, ci='sd')
        if var_name in ['Neighborhood', 'Exterior1st', 'Exterior2nd']:
            plt.sca(ax)
            plt.xticks(rotation=70) # too many variables on X axis so we need to rotate the names to that they wont interfere
        ax.set_title(var_name)
    fig.tight_layout()
    plt.savefig(cfg.visualization_dir + "\{}.png".format(filename))
    # plt.show()


def main():
    df_train = extract_data(path=cfg.train_csv_path)
    df_test = extract_data(path=cfg.test_csv_path)
    df_test.insert(df_test.shape[1] - 1, 'SalePrice', np.nan)  # need to be predicted
    df_total = pd.concat([df_train, df_test], ignore_index=True, axis=0, sort=False)
    visualizations(data_frame=df_total)
    # data_frame_train = extract_data()
    # visualizations(data_frame=data_frame_train)




def visualizations(data_frame):
    print("Features : {}".format(data_frame.columns))
    print(data_frame.info())
    # create numerical columns
    numeric_columns = data_frame.select_dtypes(include=[np.number]).columns.tolist()
    # Print how many numeric columns so figure out plot grid
    print('\n', len(numeric_columns), 'numerical columns')
    # create categorical columns
    categorical_columns = data_frame.select_dtypes(include=[np.object]).columns.tolist()
    print('\n', len(categorical_columns), 'categorical_columns')
    # plot and save distribution histograms
    draw_histograms(df=data_frame, variables=numeric_columns, n_rows=len(numeric_columns[:-1]) // 6 + 1, n_cols=6)
    # plot scatter plot w.r.t to sale price feature
    draw_scatters(df=data_frame, variables=numeric_columns, n_rows=len(numeric_columns[:-1]) // 6 + 1, n_cols=6)
    # plot bar plots w.r.t to sale price according to the categorical variables
    draw_bars(df=data_frame, variables=categorical_columns, n_rows=len(categorical_columns[:-1]) // 4 + 1, n_cols=4)
    plt.figure()
    corr = data_frame.corr()
    corr['SalePrice'].sort_values(ascending=False)
    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns,
                cmap='YlGnBu')
    plt.title('Correlation Heatmap')
    plt.savefig(cfg.visualization_dir + "\{}.png".format("Correlation"))


if __name__ == '__main__':
    main()