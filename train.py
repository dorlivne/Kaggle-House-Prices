from data_exploration import *
import os
from utils import *
from model import Benchmark_lg
from sklearn.preprocessing import StandardScaler


def split_data(X, Y, train_portion=0.5):
    msk = np.random.rand(len(X)) < train_portion
    X_train = X[msk]
    Y_train = Y[msk]
    X_val = X[~msk]
    Y_val = Y[~msk]
    return X_train, X_val, Y_train, Y_val


def monte_carlo_experiements(title, model_name, preprocessing, data,
                             results_df:pd.DataFrame, model: Benchmark_lg,
                             num_of_experiements=cfg.MC_episodes, verbose=cfg.verbose):
    X, Y = data
    mean_rmse, mean_R2_adjusted, test_mean_rmse, test_mean_R2_adjusted = 0, 0, 0, 0
    for _ in range(num_of_experiements):
        X_train, X_val, Y_train, Y_val = split_data(X, Y)  # randomly select validation set
        if cfg.scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
        model.model_fit(X_train, Y_train)
        rmse, R2_adjusted = model.model_eval(X_train, Y_train)
        mean_rmse += rmse
        mean_R2_adjusted += R2_adjusted
        test_rmse, test_R2_adjusted = model.model_eval(X_val, Y_val)
        test_mean_rmse += test_rmse
        test_mean_R2_adjusted += test_R2_adjusted
    mean_rmse /= num_of_experiements
    mean_R2_adjusted /= num_of_experiements
    test_mean_rmse /= num_of_experiements
    test_mean_R2_adjusted /= num_of_experiements
    if verbose:
        print("train: \n rmse: {}, R2_adj: {}".format(mean_rmse, mean_R2_adjusted))
        print("val: \n rmse: {}, R2_adj: {}".format(test_mean_rmse, test_mean_R2_adjusted))
    results = [build_result_line(title, 'train', model_name, preprocessing, mean_rmse, mean_R2_adjusted),
               build_result_line(title, 'val', model_name, preprocessing, test_mean_rmse, test_mean_R2_adjusted)]
    if title in results_df['title'].values:
        print("Warning: Overwriting previous experiement due to colliding title")
        user_input = over_write_or_exit_from_usr()
        if user_input == 'e':
            print("not saving results, existing....")
            exit()
        criteria = results_df['title'].values != title
        results_df = results_df[criteria]
    results_df = results_df.append(pd.DataFrame(results, columns=['title', 'dataset', 'model_name', 'preprocessing',  'rmse',  'R2_adjusted']))
    results_df.to_pickle(path=cfg.results_df)

"""
Note:
 preprocessing values in results df are described here:
normal means:
        1) clearing outliers manually via visualizations
        2) clearing outliers with Tukey's fences with k = 0.3
        3) filling missing data without deleting columns with a lot of missing data
        4) creating indicators for categorical values and ordinal values
        5) without scaling the numerical data

preprocessed 1.0 means: 
        1) ordinal values are assigned monotonically increasing value instead of indicators
        2) constant or near constant valeus are discarded - (['LowQualFinSF', 'MiscVal'])
        3) missing values in 'LotFrontage' are filled according to the neighborhood of the property for better approximation
        4) creating features from other features (feature extraction in pre-processing)
            features created:   'TotalBathAbvGrd' 'TotalBathBsmt' 'TotalRmsAbvGrdIncBath' 'YearGarageSold' 'YearRemodSold' 'TotalPorch' 'YearBuiltSold' 'YearBuiltRemod' 'ExterQualCond' 'OverallQualCond' 'GarageQualCond' 'BsmtQualCond'  
        5) SalePrice has been transformed using log1p to remove skewness and to scale the dependent variable
        6) useless features are removed according to lasso regression coeffs
        7) 1+2+3+5 from normal
"""


if __name__ == '__main__':
    if not os.path.isfile(cfg.results_df):
        results_df = pd.DataFrame()
        results_df.set_index('title')
        results_df.to_pickle(path=cfg.results_df)
    else:
        results_df = pd.read_pickle(path=cfg.results_df)
    X = pd.read_pickle(path=cfg.X_train_path)
    Y = pd.read_pickle(path=cfg.Y_train_path)
    benchmark_linear_regression = Benchmark_lg(log=True)
    monte_carlo_experiements(title='preprocessed linear regression 14.12.19', model_name='linear regression',
                             preprocessing='preprocessed 1.0',
                             data=[X, Y], results_df=results_df, model=benchmark_linear_regression)

