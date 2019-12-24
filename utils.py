from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd



def extract_data(path=r"data/train.csv"):
    data_frame_train = pd.read_csv(path)
    #  MSSubclass is assigned integers and is regarded as a numerical variable, but MSSubclass is a categorical variable
    # with no importance to order ( not ordinal like overall quality for example)
    # that means that treating mssubclass as a numerical variable is not logical because the numbers doesn't mean anything
    #  convert 'MSSubclass' column to type(str) , to be object type variable
    data_frame_train['MSSubClass'] = data_frame_train['MSSubClass'].astype(str)
    #  convert 'MoSold' to type(str) , to be object type variable
    # data_frame_train['MoSold'] = data_frame_train['MoSold'].astype(str)
    # set id to be the index of an observation and not part of the feature space
    data_frame_train.set_index('Id', inplace=True)
    return data_frame_train

def root_mean_square(predicted_value, ground_truth, log):
   if not log:
       predicted_value = np.log(predicted_value)
       ground_truth = np.log(ground_truth)
   rmse = mean_squared_error(y_true=ground_truth, y_pred=predicted_value)
   rmse = rmse ** 0.5
   return rmse


def build_result_line(title,  dataset, model_name, preprocessing, rmse, R2_adjusted ):
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
    user_input = input("Please Decide if to \n 'o' - overwrite \n 'd' - delete \n  'e' - exit")
    if user_input != 'o' and user_input != 'e' and user_input != 'd':
        print("Please choose 'e' to exit program and not save the results, 'o' to overwrite the results "
              "in the data frame or 'd' to delete the results and exit ")
        return over_write_or_exit_from_usr()
    else:
        return user_input


def find_sqrprice_by_centroid(sample_lotarea, centroids):
    """
    finding the mean SqrPrice for sample
    :param sample_lotarea: sample lot area value
    :param centroids: sorted by lot area centroids
    :return: SqrPrice of closest centroid
    """
    distance = np.inf
    ans_sqrprice = 0
    for [centroid_lotarea, centroid_sqrprice] in centroids:
        if np.abs(centroid_lotarea - sample_lotarea) < distance:
            distance = np.abs(centroid_lotarea - sample_lotarea)
            ans_sqrprice = centroid_sqrprice
        else:
            # because centroids are sorted, once the distance doesnt get shorter
            #  it will not happen with the rest of the centroids
            return ans_sqrprice
    return ans_sqrprice


