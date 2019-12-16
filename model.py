from data_exploration import *
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from utils import root_mean_square
import abc

# Note: Ridge Regression penalized the slope, theortically, we insert bias to the estimator in order to get an estimator
#       with smaller variance.
#       in practice we add a penalty to Least Squares loss function that
#       linear regression is trying to minimize, thus creating the bias so the line would not over fit to the data
#       the penalty is weighted by lambda, so that the higher lambda is the less sensitive the depended value is on
#       the independent values
#       ridge regression is really good when you don't have enough data

# Note: Lasso Regression similar to Ridge Regression. instead of using l2 penalty on the slope we use
#        l1 penalty on the slope, also adds bias to the estimator, and makes the dependent value less sensitive
#        Lasso can fix the slope to zero while Ridge can do so asymptotically
#        thus Lasso is a good method to finding useless parameters by observing their linear regression slope


class LinearModel(abc.ABC):

    @abc.abstractmethod
    def model_fit(self, X, Y, verbose=cfg.verbose):
        pass

    @abc.abstractmethod
    def model_eval(self, X, Y, verbose=cfg.verbose):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass

    @abc.abstractmethod
    def get_model(self):
        pass


class Benchmark_lg(LinearModel):

    def __init__(self, log=True,  normalize=False):
        """
        linear regression wrapper class
        :param log: if the dependent variable has already been through log
        """
        self.model = LinearRegression(normalize=normalize)
        self.log = log

    def model_fit(self, X, Y, verbose=cfg.verbose):
        self.model.fit(X, Y)
        return self

    def model_eval(self, X, Y, verbose=cfg.verbose):
        prediction_results = self.model.predict(X=X)
        rmse = root_mean_square(predicted_value=prediction_results, ground_truth=Y.values, log=self.log)
        R2_adjusted = r2_score(prediction_results, Y)
        if verbose:
            print("root-mean squared error : {}".format(rmse))
            # Explained Adjusted R^2 score: 1 is perfect prediction
            print('Variance score: {}'.format(R2_adjusted))
        return rmse, R2_adjusted

    def predict(self, X):
        return self.model.predict(X)

    def get_model(self):
        return self.model


class Ridge_Regression(Benchmark_lg):

    def __init__(self, alpha=0.1, normalize=False, log=True):
        self.model = Ridge(alpha=alpha, normalize=normalize)
        self.log = log





