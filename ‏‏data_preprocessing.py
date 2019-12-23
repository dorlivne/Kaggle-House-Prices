from data_exploration import *
from config import IowaHousePricingConfig as cfg
from sklearn.preprocessing import StandardScaler
from utils import  make_randoms
from sklearn.linear_model import Lasso
import seaborn
seaborn.set(style='ticks')


def throw_non_linear_dependent_features(df: pd.DataFrame, verbose=cfg.verbose):
    # TODO
    return df


def throw_constant_features(df: pd.DataFrame, verbose=cfg.verbose):
    """
    throw away constant(almost constant) variables
    :param df: Kaggle House sale panda dataframe
    :return:Kaggle House sale panda dataframe without constant features
    """
    if verbose:
        print("before dropping constant features, num of features is {}".format(df.shape[-1]))
    constant_features = ['LowQualFinSF', 'MiscVal']
    if verbose:
        print("dropping features: {}".format(constant_features))
    df.drop(labels=constant_features, inplace=True, axis='columns')
    if verbose:
        print("after dropping constant features, num of features is {}".format(df.shape[-1]))
    return df


def throw_non_relevent_features(df: pd.DataFrame, verbose=cfg.verbose):
    """
    throw away non-relevant variables
    :param df: Kaggle House sale panda dataframe
    :return:Kaggle House sale panda dataframe without constant features
    """
    df = throw_constant_features(df, verbose)
    df = throw_non_linear_dependent_features(df, verbose)



def fill_specific_numerical_missing_data(df, missing_numerical_columns):
    """
    through visualization we can cluster some columns by other columns and appoint more accurate values
    :param df: panda dataframe
    :param missing_numerical_columns: over all missing columns
    :return: dataframe with specific columns filled with numerical values where values were missing
    """
    specific_data = ['LotFrontage']
    df['LotFrontage'] = df['LotFrontage'].fillna(
        df.groupby(['Neighborhood'])['LotFrontage'].transform('mean'))
    missing_numerical_columns = list(filter(lambda x: x not in specific_data, missing_numerical_columns))
    return missing_numerical_columns


def fill_numerical_missing_data(df, numerical_null_column_list):
    """
    wrapper function to fill missing numerical values
    :param df: Kaggle House panda data frame
    :param numerical_null_column_list: tuples of values and columns names
    :return:
    """
    numerical_null_column_list = fill_specific_numerical_missing_data(df, numerical_null_column_list)
    # missing data on the year the garage was built, assuming that the garage was built at same time as the property
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    numerical_null_column_list.remove('GarageYrBlt')
    # all the missing numerical values are due to no basement or no masonry build  at all
    # that is why we fill them with zero because the area of a non existing basement, garage or masonry build
    # should be zero
    df[numerical_null_column_list] = df[numerical_null_column_list].fillna(0)
    # when in doubt, uncomment these comments to view that every missing value is due to non-existing bsmt, garage etc.
    # columns = df.columns
    # for col in numerical_null_column_list:
    #     null_data = df[df[col].isnull()][columns]
    #     if col == 'MasVnrArea':
    #       MasVnrType = null_data['MasVnrType']
    #     if 'Bsmt' in col:
    #         BsmtType = null_data['BsmtQual']
    #     if 'Garage' in col:
    #         GarageCond = null_data['GarageCond']
    #         detached_from_home = df.loc[df['GarageType'] == 'Detchd']
    #         detached_from_home_values = detached_from_home[col]
    return df


def fill_categorical_data(df:pd.DataFrame):
    """
    fill in missing values in categorical values
    :param df:  pd.Dataframe
    :param object_null_columns: features with null values
    :return: pd.Dataframe with no missing values
    """
    # Non existing SaleType parameters are filled with Other
    df['SaleType'] = df['SaleType'].fillna('Oth')
    # Non existing Utilities parameters are filled with Electricity only
    df['Utilities'] = df['Utilities'].fillna('ELO')
    # Non existing categorical non-ordinal parameters are filled Maximum a-prior value
    features = ['MSZoning', 'Exterior1st', 'Exterior2nd', 'Electrical', 'Functional']
    for feature in features: df[feature] = df[feature].fillna(
        df[feature].value_counts().idxmax())  # estimate missing values with MAP
    # Non existing MasVnrType parameters that have area in feature data are filled with  Maximum a-prior value
    df['MasVnrType'][df['MasVnrArea'] != 0] = df['MasVnrType'].fillna(
        df['MasVnrType'].value_counts().idxmax())  # estimate missing values with MA
    # Non existing MasVnrType parameters are filled with None
    df['MasVnrType'] = df.fillna('None')
    # Non existing Quality/Condition/Other parameters are filled with NA
    features = ['Fence', 'MiscFeature', 'Alley', 'GarageType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                'BsmtFinType2', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC', 'GarageFinish']
    df[features] = df[features].fillna('NA')
    return df


def clear_outliers(df: pd.DataFrame, k=cfg.K_Tukey_far_out, visualize=cfg.visualize):
    """
    clear outliers via Tukey's fences
    :param df: panda data frame
    :param k: determines the tolerance for outliers, k ~ tolerance
    :param visualize: visualize the box plots before and after throwing away outliers
    :return: pd.DataFrame with no outliers according to Tukey's fences criterion
    """
    if visualize:
        plt.figure()
        sns.boxplot(df['SalePrice'])
        plt.title('SalePrice Original')
        plt.savefig(cfg.visualization_dir + "/SalePriceBoxPlotWithOutliers")
    statistics_of_df = df['SalePrice'].describe()
    # Tukey's fences determine what is an outlier with percentiles
    # view https://en.wikipedia.org/wiki/Outlier#Definitions_and_detection
    Q1_25 = statistics_of_df.get('25%')
    Q1_75 = statistics_of_df.get('75%')
    lower_bound = Q1_25 - k * (Q1_75 - Q1_25)
    upper_bound = Q1_75 + k * (Q1_75 - Q1_25)
    criteria = (lower_bound <= df['SalePrice']) & (df['SalePrice'] <= upper_bound)  # not outliers according to Tukey's fences
    df_without_outliers = df[criteria]
    if visualize:
        plt.figure()
        sns.boxplot(df_without_outliers['SalePrice'])
        plt.title("SalePrice after removing outilers with Tukey's  fences method")
        plt.savefig(cfg.visualization_dir + "/SalePriceBoxPlotWithoutOutliers")
    return df_without_outliers


def clear_outliers_manually(df: pd.DataFrame):
    """
    clear samples from data frame by visualizations, hand picked anomalies
    :param df: data frame
    :return: the same data frame, minus specific samples that were determines as bad samples in the data exploration process
    """
    criteria = df['GrLivArea'] >= 3500  # view scatter_plot graph, values above 3500 are anomalies in the data
    samples_from_GRLivArea = df[criteria]
    df = df.drop(samples_from_GRLivArea.index)
    # draw_scatters(df, ['GrLivArea'], n_rows=1, n_cols=1, filename='tmp_aftrer') # doubled check that anomalies got off
    criteria = (df['TotalBsmtSF'] >= 3000) * (df['SalePrice'] <= 300000)  # view scatter_plot graph, values above 3500 are anomalies in the data
    samples_from_TotalBsmtSF = df[criteria]
    df = df.drop(samples_from_TotalBsmtSF.index)
    # draw_scatters(df, ['TotalBsmtSF'], n_rows=1, n_cols=1, filename='TotalBsmtSF_no_anomalies') # doubled check that anomalies got off
    criteria = df['LotArea'] >= 100000
    samples_from_LotArea = df[criteria]
    df = df.drop(samples_from_LotArea.index)
    criteria = df['OpenPorchSF'] >= 500
    samples_from_OpenPorchSF = df[criteria]
    df = df.drop(samples_from_OpenPorchSF.index)
    # draw_scatters(df, ['OpenPorchSF'], n_rows=1, n_cols=1, filename='OpenPorchSF_no_anomalies') # doubled check that anomalies got off
    return df


def fill_missing_data(df: pd.DataFrame, verbose=cfg.verbose):
    """
    wrapper function to fill all the missing values in the data
    :param df: panda data frame holding the Kaggle House sale data
    :param verbose: to print or not to print
    :return:  no missing values panda data frame, there should be no null columns when this function ends
    """
    object_null_column_list = []
    numerical_null_column_list = []
    categorical_columns = df.select_dtypes(include=[np.object]).columns.tolist()
    # Find how many missing values and percentage missing in each column
    # This gives me an idea of priority
    if verbose:
        print("list of null columns {}"
              .format(df.columns[df.isnull().any()].tolist()))
        print('Missing data in each dataframe column:')
    num_of_samples = len(df)
    for col in df.columns:
        missing_data = len(df) - df[col].count()
        if (missing_data > 0 or missing_data == 'NaN'):
            precentage_missing = round(missing_data / num_of_samples * 100, 3)
            if verbose:
                print(col, ':', missing_data, 'missing values is',
                      str(precentage_missing), '% of total')
            if col in categorical_columns:
                object_null_column_list.append(col)
            else:
                numerical_null_column_list.append(col)
    # fill numerical values to missing values in numerical columns
    df = fill_numerical_missing_data(df, numerical_null_column_list)
    # fill categorical values
    df = fill_categorical_data(df)
    if verbose:
        null_columns = df.columns[df.isnull().any()].tolist()
        assert null_columns == [], " {} has null values, cannot proceed".format(null_columns)
    return df


def create_quality_area(quality, area, df: pd.DataFrame, drop_quality=True, drop_area=True):
    values = set(df[quality].values)
    for value in values:
        criteria = df[quality] == value
        df['{}_{}'.format(area, value)] = 0
        df.loc[criteria, '{}_{}'.format(area, value)] = df[area]
    if drop_area:
        df.drop([area], axis='columns', inplace=True)
    if drop_quality:
        df.drop([quality], axis='columns', inplace=True)
    return df

def feature_extraction(df:pd.DataFrame, verbose=cfg.verbose):
    """
    manually preform feature extraction
    :param df: Kaggle House Sale Price data frame
    :return: Kaggle House Sale Price data frame with engineered features
    """
    if verbose:
        print("Creating features from existing featurese")
    # feature extraction: features related to room
    df['TotalBathAbvGrd'] = df['FullBath'] + 0.5 * df['HalfBath']
    df['TotalBathBsmt'] = df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    df['TotalRmsAbvGrdIncBath'] = df['TotRmsAbvGrd'] + df['TotalBathAbvGrd']
    df = df.drop(['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'], axis='columns')
    # df['TotalRms'] = df['TotalRmsAbvGrdIncBath'] + df['TotalBathBsmt'] # low LASSO coeff

    # feature extraction: features related to area
    # df['TotalSF'] = df['TotalBsmtSF'] + df['GrLivArea'] # low LASSO coeff
    df['TotalPorch'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['3SsnPorch']
    # df = df.drop(['OpenPorchSF', 'EnclosedPorch', 'ScreenPorch', '3SsnPorch'], axis='columns')
    # feature extraction: assigning number to ordinal features
    ordinal_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                        'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
    df[ordinal_features] = df[ordinal_features].replace('Ex', 5).\
                           replace('Gd', 4).replace('TA', 3).replace('Fa', 2).replace('Po', 1).replace('NA', 0)
    # creating Quality * Cond features
    # Example: if the condition and the quality of the bsmt is high the weight of the BsmtQualCond is 25!
    df['ExterQualCond'] = df['ExterQual'] * df['ExterCond']
    df['BsmtQualCond'] = df['BsmtQual'] * df['BsmtCond']
    df['GarageQualCond'] = df['GarageQual'] * df['GarageCond']
    df['OverallQualCond'] = df['OverallQual'] * df['OverallCond']
    # df = df.drop(['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'OverallQual', 'OverallCond'], axis='columns')
    # feature extraction: assigning number to ordinal features
    ordinal_features = ['BsmtExposure']
    df[ordinal_features] = df[ordinal_features].replace('Gd', 4).replace('Av', 3).replace('Mn', 2).replace('No', 1)\
        .replace('NA', 0)
    # feature extraction: assigning number to ordinal features
    ordinal_features = ['BsmtFinType1']
    df[ordinal_features] = df[ordinal_features].replace('GLQ', 6).replace('ALQ', 5).replace('BLQ', 4).replace('Rec', 3)\
        .replace('LwQ', 2).replace('Unf', 1).replace('NA', 0)
    # feature extraction: assigning number to ordinal features
    ordinal_features = ['GarageFinish']
    df[ordinal_features] = df[ordinal_features].replace('Fin', 3).replace('RFn', 2).replace('Unf', 1).replace('NA', 0)
    # feature extraction: assigning number to ordinal features
    ordinal_features = ['Fence']
    df[ordinal_features] = df[ordinal_features].replace('GdPrv', 4).replace('MnPrv', 3).replace('GdWo', 2)\
        .replace('MnWw', 1).replace('NA', 0)
    # creating time features
    df['YearBuiltRemod'] = df['YearRemodAdd'] - df['YearBuilt']
    df['YearBuiltSold'] = df['YrSold'] - df['YearBuilt']
    df['YearRemodSold'] = df['YrSold'] - df['YearRemodAdd']
    df['YearGarageSold'] = df['YrSold'] - df['GarageYrBlt']
    df['HasPool'] = df['PoolArea'] > 0
    df = df.drop(['PoolQC', 'PoolArea'], axis='columns')
    # df['PoolQcArea'] = df['PoolQC'] * df['PoolArea']
    # df = create_quality_area(quality='MasVnrType', area='MasVnrArea', df=df)
    # df = create_quality_area(quality='BsmtFinType1', area='BsmtFinSF1', df=df)
    # df = create_quality_area(quality='GarageQual', area='GarageArea', df=df)
    # df = create_quality_area(quality='PoolQC', area='PoolArea', df=df)
    # df.drop(['BsmtFinType2', 'BsmtFinSF2', 'ExterCond', '3SsnPorch',
    #          'ScreenPorch', 'YrSold', 'MoSold', 'OverallCond', 'PoolQC', 'PoolArea'], axis='columns', inplace=True)
    return df


def fixing_skewness(df:pd.DataFrame, visualize=False):
    """
    manually preform feature engineering
    :param df: Kaggle House Sale Price data frame
    :return: Kaggle House Sale Price data frame with engineered features
    """
    # skewness to make the data normal distributed
    numerical_columns = df.select_dtypes(include=['number']).columns.drop(['SalePrice']).tolist()
    # already transformed SalePrice to be normalized
    for col in numerical_columns:
        values_set = df[col].values
        values_set = set(values_set)
        length_of_set = len(values_set)
        if length_of_set < 20:
            print("{} is sparse, we didnt skew it".format(col))
            continue
        data = df[col]
        skewness = data.skew()
        if skewness > 0.75:
            # TODO: should I transform features with a majority of zeroes?
            if visualize:
                draw_histograms(df=df, variables=[col], n_cols=1, n_rows=1, save_fig=False, show_fig=True, fig_size=None)
            df[col] = np.log1p(df[col])  # normalize the data
            if visualize:
                draw_histograms(df=df, variables=[col], n_cols=1, n_rows=1, save_fig=False, show_fig=True, fig_size=None)
    return df

def prune_features(X_train, Y_train , verbose=cfg.verbose):
    """
    finding useless variables using lasso regression
    :param X: features dataframe
    :param Y: dependet variable
    :param verbose: to print or not to print
    :return:useless and usefull features names according to lasso regression
    """

    lasso = Lasso(alpha=1e-4).fit(X=X_train, y=Y_train)
    df_features_score = pd.DataFrame(data=lasso.coef_, index=X_train.columns, columns=['features_score'])
    # remove the useless features from test and train
    usefull_features_criteria = (df_features_score.values <= -5e-5) + (df_features_score.values >= 5e-5)
    usefull_features = df_features_score[usefull_features_criteria]
    useless_features = df_features_score[[not i for i in usefull_features_criteria]]
    useless_features = useless_features.sort_values('features_score')
    usefull_features = usefull_features.sort_values('features_score')
    usefull_features_names = usefull_features.index
    useless_features_names = useless_features.index
    if cfg.verbose:
        print("removed features and their importance according to LASSO coeff:")
        for feature in useless_features_names:
            print("feature {} : {}".format(feature, df_features_score['features_score'].loc[feature]))
        print("usefull features and their importance according to LASSO coeff:")
        for feature in usefull_features_names:
            print("feature {} : {}".format(feature, df_features_score['features_score'].loc[feature]))
    return usefull_features_names, useless_features_names

def get_rid_of_sparse_columns(df:pd.DataFrame, verbose=cfg.verbose):
    columns_to_throw_out = []
    for col in df.columns:
        zeros = df[col].value_counts().iloc[0]
        if (zeros / len(df)) * 100 > 99.94:  # 99.94 is according to Sandeep Kumar's work
            columns_to_throw_out.append(col)
    if verbose:
        print("sparse columns {}".format(columns_to_throw_out))
    df = df.drop(columns_to_throw_out, axis='columns')
    return df

def drop_columns_by_correlation(corr_list:pd.DataFrame, df:pd.DataFrame):

    criteria = (-0.02 < corr_list['SalePrice']) * (corr_list['SalePrice'] < 0.02)
    uncorrelated_columns = df[criteria.index[criteria]]
    df.drop(uncorrelated_columns, axis='columns', inplace=True)
    return df


def preprocessing():
    print("------------------ Extracting data from CSV files ------------------")
    df_train = extract_data(path=cfg.train_csv_path)
    # extracting test data frame from csv file
    df_test = extract_data(path=cfg.test_csv_path)
    # remove outliers from train
    print("------------------ Removing outliers from train ------------------")
    df_train = clear_outliers_manually(df_train)
    df_train = clear_outliers(df=df_train)
    # rearrange index because we deleted samples
    df_train.reset_index(inplace=True)
    df_train.drop(['Id'], axis=1, inplace=True)
    print("------------------ Concatenating train and test to one Data Frame ------------------")
    # adding the SalePrice dependent variable so we can concatenate test and train for preprocessing purposes
    df_test.insert(df_test.shape[1] - 1, 'SalePrice', np.nan)  # need to be predicted
    df_train['DataType'], df_test['DataType'] = 'train', 'test'  # to divide them later on
    # treat all the data as one, so we would preform the same changes on test and on train
    df_total = pd.concat([df_train, df_test], ignore_index=True, axis=0, sort=False)

    print("------------------ Scaling the dependent variable with log1p transform ------------------")
    df_total['SalePrice'] = np.log1p(df_total['SalePrice'])  # natural logarithm to fix the scale of SalePrices
    print("------------------ Throwing out constant features ------------------")
    df_total = throw_constant_features(df_total)
    print("------------------ Filling missing data ------------------")
    df_total = fill_missing_data(df_total)
    print("------------------ Fixing skewness in data ------------------")
    df_total = fixing_skewness(df_total)
    print("------------------ Feature extraction ------------------")
    df_total = feature_extraction(df_total)
    print("------------------  Transforming categorical values to indicator features ------------------")
    # transforming categorical values to indicator features
    df_total = pd.get_dummies(df_total, columns=None, drop_first=True)
    print("------------------ Dropping sparse features ------------------")
    df_total = get_rid_of_sparse_columns(df_total)
    print("------------------  Seperating back to test and train data frames ------------------")
    df_test = df_total[df_total['DataType_train'] == 0]
    df_train = df_total[df_total['DataType_train'] == 1]
    df_test = df_test.drop(['DataType_train', 'SalePrice'], axis='columns')
    df_train = df_train.drop('DataType_train', axis='columns')
    print("------------------  Dividing train to independent & dependent variables ------------------")
    X_train = df_train.drop(['SalePrice'], axis='columns')
    Y_train = df_train['SalePrice']
    # print("------------------  Pruning features with Lasso Regression ------------------")
    usefull_features, _ = prune_features(X_train, Y_train)
    X_train = X_train[usefull_features]
    df_test = df_test[usefull_features]
    print("------------------  Saving Train and Test data frames ------------------")
    pd.to_pickle(X_train, cfg.X_train_path)
    pd.to_pickle(Y_train, cfg.Y_train_path)
    pd.to_pickle(df_test, cfg.X_test_path)


if __name__ == '__main__':
    preprocessing()

