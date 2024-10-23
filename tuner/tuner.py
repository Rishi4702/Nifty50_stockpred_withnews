import tensorflow as tf
from keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
import matplotlib.pyplot as plt

class LSTMModelTuner:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.tuner = self.build_tuner()

    def build_model(self, hp):
        model = Sequential()
        model.add(LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32),
                       return_sequences=True,
                       input_shape=(self.x_train.shape[1], self.x_train.shape[2])))
        model.add(Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
        model.add(LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32)))
        model.add(Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
        model.add(Dense(1, activation='linear'))

        model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                      loss='mean_squared_error')

        return model

    def build_tuner(self):
        tuner = kt.RandomSearch(
            self.build_model,
            objective='val_loss',
            max_trials=10,
            executions_per_trial=2,
            directory='lstm_tuner',
            project_name='ohcl_greed_fear'
        )
        return tuner

    def search_best_hyperparameters(self):
        self.tuner.search(self.x_train, self.y_train,
                          epochs=10,
                          validation_data=(self.x_test, self.y_test),
                          callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])
        best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        return best_hps

    def get_best_model(self):
        best_hps = self.search_best_hyperparameters()
        model = self.build_model(best_hps)
        return model

    def plot_trial_metrics(self):
        trials = self.tuner.oracle.get_best_trials(num_trials=10)
        val_losses = [trial.metrics.get_best_value('val_loss') for trial in trials]
        trial_ids = [trial.trial_id for trial in trials]
        sorted_trials = sorted(zip(trial_ids, val_losses, trials), key=lambda x: x[1])

        trial_ids_sorted = [x[0] for x in sorted_trials]
        val_losses_sorted = [x[1] for x in sorted_trials]
        trials_sorted = [x[2] for x in sorted_trials]

        min_loss = min(val_losses_sorted)
        max_loss = max(val_losses_sorted)
        padding = (max_loss - min_loss) * 0.1  # Add 10% padding to y-axis range

        # Plot validation loss
        plt.figure(figsize=(12, 6))
        plt.bar(trial_ids_sorted, val_losses_sorted, color='blue')
        plt.title('Validation Loss for Each Trial')
        plt.xlabel('Trial ID')
        plt.ylabel('Validation Loss')
        plt.xticks(rotation=45)
        plt.ylim(min_loss - padding, max_loss + padding)
        plt.grid(True)
        plt.show()

        # Plot hyperparameters
        units = [trial.hyperparameters.values['units'] for trial in trials_sorted]
        dropouts = [trial.hyperparameters.values['dropout'] for trial in trials_sorted]
        learning_rates = [trial.hyperparameters.values['learning_rate'] for trial in trials_sorted]

        fig, axs = plt.subplots(3, 1, figsize=(12, 18))

        axs[0].bar(trial_ids_sorted, units, color='blue')
        axs[0].set_title('Units for Each Trial')
        axs[0].set_xlabel('Trial ID')
        axs[0].set_ylabel('Units')
        axs[0].set_xticks(range(len(trial_ids_sorted)))
        axs[0].set_xticklabels(trial_ids_sorted, rotation=45)
        axs[0].grid(True)

        axs[1].bar(trial_ids_sorted, dropouts, color='green')
        axs[1].set_title('Dropout Rate for Each Trial')
        axs[1].set_xlabel('Trial ID')
        axs[1].set_ylabel('Dropout Rate')
        axs[1].set_xticks(range(len(trial_ids_sorted)))
        axs[1].set_xticklabels(trial_ids_sorted, rotation=45)
        axs[1].grid(True)

        axs[2].bar(trial_ids_sorted, learning_rates, color='red')
        axs[2].set_title('Learning Rate for Each Trial')
        axs[2].set_xlabel('Trial ID')
        axs[2].set_ylabel('Learning Rate')
        axs[2].set_xticks(range(len(trial_ids_sorted)))
        axs[2].set_xticklabels(trial_ids_sorted, rotation=45)
        axs[2].set_yscale('log')  # Use logarithmic scale for learning rates
        axs[2].grid(True)

        for ax, param in zip(axs, [units, dropouts, learning_rates]):
            for idx, val in enumerate(param):
                if isinstance(val, float):
                    ax.text(idx, val, f'{val:.2e}', ha='center', va='bottom', fontsize=8)
                else:
                    ax.text(idx, val, f'{val}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.show()

        # Display hyperparameters of top trials
        for i, (trial_id, val_loss, trial) in enumerate(sorted_trials[:5]):
            print(f"Trial ID: {trial_id} - Validation Loss: {val_loss}")
            print(f"Hyperparameters: {trial.hyperparameters.values}\n")
