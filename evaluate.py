from data_exploration import *
import os
from utils import *
from model import Benchmark_lg
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    if not os.path.isfile(cfg.results_df):
        results_df = pd.DataFrame()
        results_df.set_index('title')
        results_df.to_pickle(path=cfg.results_df)
    else:
        results_df = pd.read_pickle(path=cfg.results_df)
    X = pd.read_pickle(path=cfg.X_train_path)
    Y = pd.read_pickle(path=cfg.Y_train_path)
    X_test = pd.read_pickle(path=cfg.X_test_path)
    model = Benchmark_lg(log=True)
    model.model_fit(X, Y)
    predictions = model.predict(X_test)
    raw_predictions = np.exp(predictions) - 1
    fig = plt.figure()
    plt.hist(raw_predictions)
    plt.show()
    results = pd.DataFrame(np.arange(start=1461, stop=2920), columns=['Id'])
    results['SalePrice'] = raw_predictions
    results.to_csv(path_or_buf=cfg.results_for_submission_path, index=False)