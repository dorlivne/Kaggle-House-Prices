from data_exploration import *
import os
from utils import *
from model import Benchmark_lg, Ridge_Regression, LinearModel, Ensemble_reg, Elastic_Regression, Kernal_Regression
from sklearn.model_selection import cross_val_score, GridSearchCV

def grid_search(params_grid, X, Y, linear_model:LinearModel):
    linear_model_cv = GridSearchCV(linear_model.get_model(), params_grid, iid=False, cv=5)
    linear_model_cv.fit(X, Y)
    print("Best parameters set found on development set:")
    print()
    print(linear_model_cv.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = linear_model_cv.cv_results_['mean_test_score']
    stds = linear_model_cv.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, linear_model_cv.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    return linear_model_cv.best_params_


def evaluate(model, X_test, X, Y):
    model.model_fit(X, Y)
    print("------------------  SalePrice evaluation on Test set ------------------")
    predictions = model.predict(X_test)
    print("------------------  Assuming transformation of SalePrice ------------------")
    print("------------------ transforming back to raw SalePrice values ------------------ ")
    raw_predictions = np.exp(predictions) - 1
    print("------------------ Plotting bar-plot of results, use this to check if results make sense ------------------")
    fig = plt.figure()
    plt.hist(raw_predictions)
    plt.show()
    print("------------------ Saving results in path ------------------")
    results = pd.DataFrame(np.arange(start=1461, stop=2920), columns=['Id'])
    results['SalePrice'] = raw_predictions
    results.to_csv(path_or_buf=cfg.results_for_submission_path, index=False)


if __name__ == '__main__':
    if not os.path.isfile(cfg.results_df):
        results_df = pd.DataFrame()
        results_df.set_index('title')
        results_df.to_pickle(path=cfg.results_df)
    else:
        results_df = pd.read_pickle(path=cfg.results_df)
    print("------------------  Reading Train and Test ------------------")
    X = pd.read_pickle(path=cfg.X_train_path)
    Y = pd.read_pickle(path=cfg.Y_train_path)
    X_test = pd.read_pickle(path=cfg.X_test_path)
    print("------------------  Loading Model ------------------")
    model = Ridge_Regression(alpha=10.0)
    print("------------------  Training Model on Train set ------------------")
    evaluate(model, X_test, X, Y)