
class IowaHousePricingConfig:
    visualization_dir = r"C:\Study\סמסטר ח\dataMining\Assignment2\visualizations"
    model_dir = r"C:\Study\סמסטר ח\dataMining\Assignment2\saved_models"
    K_Tukey_outlier = 1.5
    K_Tukey_far_out = 3
    visualize = False
    verbose = True
    train_csv_path = r"data/train.csv"
    test_csv_path = r"data/test.csv"
    all_csv_path = r"data/all.csv"
    X_train_path = r"data/X.pkl"
    Y_train_path = r"data/Y.pkl"
    X_test_path = r"data/X_test.pkl"
    regression_coeffs_path = r"model/linear_regression/coeffs"
    MC_episodes = 10  # arbitrary number
    results_df = r"saved_models/results.pkl"
    results_path = r"saved_models/results_models.csv"
    results_for_submission_path = r"results.csv"
    scale = True