from tuner.tuner import *
from Dataprocessor import *


def merge_csv_files(file_path1):
    """
    Merges two CSV files based on their date index and identifies unmerged dates.

    Parameters:
    - file_path1: str, the file path to the first CSV file.
    - file_path2: str, the file path to the second CSV file.

    Returns:
    - merged_df: DataFrame, the resulting merged DataFrame.
    - unmerged_dates: DataFrame, rows with dates that did not merge in both files.
    """
    file_path2 =r'C:\Users\golur\PycharmProjects\NIfty50_stockpred\dataset\upload_DJIA_table.csv'
    # Load the CSV files
    df1 = pd.read_csv(file_path1, index_col='Date', parse_dates=True)
    df2 = pd.read_csv(file_path2, index_col='Date', parse_dates=True)

    # Merge the DataFrames on the date column
    merged_df = pd.merge(df1, df2, on='Date', how='outer', indicator=True)

    # Reset index to make 'Date' a column
    merged_df.reset_index(inplace=True)

    # Print merged dataframe head
    print(merged_df.head())

    # Select specific columns
    columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Class_1_Percentage']
    df_selected = merged_df[columns]
    df_selected.to_csv('normalbert.csv', index=False)


# Entry point for running the stock prediction
if __name__ == "__main__":
    data_loader_finbert = DataLoader('./finbert.csv')
    data_loader_bert = DataLoader('./normalbert.csv')
    data_loader_roberta = DataLoader('./roberta.csv')
    x_train, y_train, x_test, y_test = data_loader.x_train, data_loader.y_train, data_loader.x_test, data_loader.y_test

    tuner = LSTMModelTuner(x_train, y_train, x_test, y_test)
    best_model = tuner.get_best_model()

    # Train the best model
    best_model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test),
                   callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])

    # Plot the trial metrics
    tuner.plot_trial_metrics()
