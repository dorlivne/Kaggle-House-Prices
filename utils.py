from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd


def root_mean_square(predicted_value, ground_truth, log):
   if not log:
       predicted_value = np.log(predicted_value)
       ground_truth = np.log(ground_truth)
   rmse = mean_squared_error(y_true=ground_truth, y_pred=predicted_value)
   rmse = rmse ** 0.5
   return rmse


def build_result_line(title,  dataset, model_name, preprocessing, rmse, R2_adjusted):
 return {'title': title, 'dataset': dataset, 'model_name': model_name,
          'preprocessing': preprocessing, 'rmse': rmse, 'R2_adjusted': R2_adjusted}


def make_randoms(df, col: pd.Series):
    """
    function to make random distribution based on panda series data
    :param df: data frame
    :param col: specific column
    :return: sampled value from column distribution
    """
    return np.random.normal(loc=df[col].mean(),
                            scale=df[col].std(),
                            size=df[col].isnull().sum()).astype(int)


def over_write_or_exit_from_usr():
    user_input = input("Please Decide if to \n 'o' - overwrite \n 'e' - exit \n ")
    if user_input != 'o' and user_input != 'e':
        print("Please choose 'e' to exit program and not save the results or 'o' to overwrite the results in the data frame ")
        return over_write_or_exit_from_usr()
    else:
        return user_input