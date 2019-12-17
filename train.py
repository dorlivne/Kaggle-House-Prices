from data_exploration import *
import os
from utils import *
from model import Benchmark_lg, LinearModel, Ridge_Regression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score, GridSearchCV

def split_data(X, Y, train_portion=0.75):
    msk = np.random.rand(len(X)) < train_portion
    X_train = X[msk]
    Y_train = Y[msk]
    X_val = X[~msk]
    Y_val = Y[~msk]
    return X_train, X_val, Y_train, Y_val


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


def evaluate_model_on_train(title, model_name, preprocessing, data,
                            results_df:pd.DataFrame, model: LinearModel, verbose=cfg.verbose):
    """
    Cross-validation, split train data frame to equally sized val and train data sets and evaluate R2 adjusted and RMSE
    :param title: name of experiment to be saved in results data frame
    :param model_name: name of linear model used in experiment
    :param preprocessing: type of postprocessing, if a massive change is preformed please write down in short
                          what the postprocessing includes(like normal)
    :param data: train data frame with SalePrice given
    :param results_df: where to put the result of the experiment
    :param model: model that implements the LinearModel interface
    :param verbose: to print the results on console
    :return:
    """
    print("------------------  Training on train and CV on Train ------------------")
    X, Y = data
    if cfg.scale:
        print("------------------  Scaling Train ------------------")
        scaler = RobustScaler()
        X = scaler.fit_transform(X)

    model.model_fit(X, Y)
    print("------------------  Evaluating on Train ------------------")
    train_rmse, train_R2_adjusted = model.model_eval(X, Y, False)
    rmse_cv_score = np.sqrt(np.abs(cross_val_score(model.get_model(), X, Y, cv=5, scoring='neg_mean_squared_error')))
    R2_cv_score = cross_val_score(model.get_model(), X, Y, cv=5, scoring='r2')
    if verbose:
        print("train: \n rmse: {}, R2_adj: {}".format(train_rmse, train_R2_adjusted))
        print("cross validation rmse score: {} (+/- {}".format(rmse_cv_score.mean(), rmse_cv_score.std() * 2))
        print("cross validation r2 score: {} (+/- {}".format(R2_cv_score.mean(), R2_cv_score.std() * 2))
    results = [build_result_line(title, 'train', model_name, preprocessing, train_rmse, train_R2_adjusted),
               build_result_line(title, 'val', model_name, preprocessing, rmse_cv_score.mean(), R2_cv_score.mean())]
    if title in results_df['title'].values:
        print("Warning: Overwriting previous experiement due to colliding title")
        user_input = over_write_or_exit_from_usr()
        if user_input == 'e':
            print("not saving results, existing....")
            exit()
        if user_input == 'd':
            print("deleting previous results, existing....")
            criteria = results_df['title'].values != title
            results_df = results_df[criteria]
            results_df.to_pickle(path=cfg.results_df)
            results_df.to_csv(path_or_buf=cfg.results_path, index=False)
            exit()
        criteria = results_df['title'].values != title
        results_df = results_df[criteria]
    results_df = results_df.append(pd.DataFrame(results, columns=['title', 'dataset', 'model_name', 'preprocessing',  'rmse',  'R2_adjusted']))
    results_df.to_pickle(path=cfg.results_df)
    results_df.to_csv(path_or_buf=cfg.results_path, index=False)


if __name__ == '__main__':
    if not os.path.isfile(cfg.results_df):
        results_df = pd.DataFrame()
        results_df.set_index('title')
        results_df.to_pickle(path=cfg.results_df)
    else:
        results_df = pd.read_pickle(path=cfg.results_df)

    print("------------------  Reading Train from path ------------------")
    X = pd.read_pickle(path=cfg.X_train_path)
    Y = pd.read_pickle(path=cfg.Y_train_path)
    linear_model = Ridge_Regression(alpha=10.0)
    # alpha_range = np.logspace(-4, 4, base=10, num=9)
    # params_grid = {'alpha': alpha_range}
    # linear_model_cv = GridSearchCV(linear_model.get_model(), params_grid, iid=False, cv=5)
    # linear_model_cv.fit(X, Y)
    # print("Best parameters set found on development set:")
    # print()
    # print(linear_model_cv.best_params_)
    # print()
    # print("Grid scores on development set:")
    # print()
    # means = linear_model_cv.cv_results_['mean_test_score']
    # stds = linear_model_cv.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, linear_model_cv.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean, std * 2, params))
    # print()

    evaluate_model_on_train(title='preprocessed + normalized ridge regression 17.12.19', model_name='ridge regression',
                            preprocessing='preprocessed 1.0 + dropping features during extraction',
                            data=[X, Y], results_df=results_df, model=linear_model)



