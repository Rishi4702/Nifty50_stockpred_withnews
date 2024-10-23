import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DataLoader:
    def __init__(self, data_path, sequence_length=50, train_split=0.8):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.train_split = train_split
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = self.load_data()
        self.x_train, self.y_train, self.x_test, self.y_test = self.preprocess_data()

    def load_data(self):
        df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        return df

    def normalize_data(self, data):
        return self.scaler.fit_transform(data)

    def preprocess_data(self):
        data = self.data[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Class_1_Percentage']]
        normalized_data = self.normalize_data(data)
        x_data = []
        y_data = []
        for i in range(len(normalized_data) - self.sequence_length):
            x_data.append(normalized_data[i:i + self.sequence_length, :-1])
            y_data.append(normalized_data[i + self.sequence_length, -1])
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        split_index = int(len(x_data) * self.train_split)
        x_train = x_data[:split_index]
        y_train = y_data[:split_index]
        x_test = x_data[split_index:]
        y_test = y_data[split_index:]
        return x_train, y_train, x_test, y_test


def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.4))
    model.add(LSTM(128))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001382765463625165), loss='mean_squared_error')
    return model


def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    return mse, mae, rmse, r2


def plot_metrics(metrics, datasets, metric_names):
    num_metrics = len(metric_names)
    fig, axs = plt.subplots(num_metrics, 1, figsize=(10, num_metrics * 5))

    for i, metric_name in enumerate(metric_names):
        values = [metrics[j][i] for j in range(len(datasets))]
        axs[i].bar(datasets, values, color='blue')
        axs[i].set_title(f'Model {metric_name} on Different Datasets')
        axs[i].set_xlabel('Dataset')
        axs[i].set_ylabel(metric_name)
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    datasets = ['finbert.csv', 'normalbert.csv', 'roberta.csv']
    all_metrics = []

    for dataset in datasets:
        data_loader = DataLoader(f'./{dataset}')
        model = build_model(input_shape=(data_loader.x_train.shape[1], data_loader.x_train.shape[2]))
        model.fit(data_loader.x_train, data_loader.y_train, epochs=50,
                  validation_data=(data_loader.x_test, data_loader.y_test),
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])
        metrics = evaluate_model(model, data_loader.x_test, data_loader.y_test)
        all_metrics.append(metrics)

    metric_names = ['MSE', 'MAE', 'RMSE', 'R2']
    plot_metrics(all_metrics, datasets, metric_names)
