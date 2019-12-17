from data_exploration import *
import os
from utils import *
from model import Benchmark_lg, Ridge_Regression

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