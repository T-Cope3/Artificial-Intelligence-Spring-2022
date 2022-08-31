# This is a sample Python script.
# import warnings
# warnings.filterwarnings("ignore")

# Programming Assignment-01-CS4267-Troy-Cope

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
from sklearn import preprocessing, datasets


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# Activation funtion, good way to keep values low for computation
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1.0 - sigmoid(z))


# We then define training fuction which takes
# training dataset and num of units in hidden laryer
# and num of iterations

# straight forward, training the network through back propagation and matrix math (matmul)
def train(X, y, n_hidden, learning_rate, n_iter):
    m, n_input = X.shape
    W1 = np.random.randn(n_input, n_hidden)
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, 1)
    b2 = np.zeros((1, 1))

    for i in range(1, n_iter + 1):
        Z2 = np.matmul(X, W1) + b1
        A2 = sigmoid(Z2)
        Z3 = np.matmul(A2, W2) + b2
        A3 = Z3
        dZ3 = A3 - y
        dW2 = np.matmul(A2.T, dZ3)
        db2 = np.sum(dZ3, axis=0, keepdims=True)
        dZ2 = np.matmul(dZ3, W2.T) * sigmoid_derivative(Z2)
        dW1 = np.matmul(X.T, dZ2)
        db1 = np.sum(dZ2, axis=0)
        W2 = W2 - learning_rate * dW2 / m
        b2 = b2 - learning_rate * db2 / m
        W1 = W1 - learning_rate * dW1 / m
        b1 = b1 - learning_rate * db1 / m

        if i % 100 == 0:
            cost = np.mean((y - A3) ** 2)
            print('Iteration %i, training loss: %f' % (i, cost))

    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model


# reading in the data from the boston set
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
boston_data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
boston_target = raw_df.values[1::2, 2]

# boston = datasets.load_boston()

# need to fix the datasets
num_test = 10

scaler = preprocessing.StandardScaler()
X_train = boston_data[:-num_test, :]
X_train = scaler.fit_transform(X_train)
y_train = boston_target[:-num_test].reshape(-1, 1)
X_test = boston_data[-num_test:, :]
X_test = scaler.transform(X_test)
y_test = boston_target[-num_test:]

# With the scaled dataset, we can now train a one-layer neural network with 20 hidden
# units, a 0.1 learning rate, and 2000 iterations:

# manually declaring the values for training
n_hidden = 20
learning_rate = 0.1
n_iter = 2000
model = train(X_train, y_train, n_hidden, learning_rate, n_iter)


# a prediction function to use the values from the model
def predict(x, model):
    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']
    A2 = sigmoid(np.matmul(x, W1) + b1)
    A3 = np.matmul(A2, W2) + b2
    return A3


# finds the data test vs the current model to check accuracy
predictions = predict(X_test, model)

# print predictions and ground truths to compare
print(predictions)
print(y_test)

# different way to train and check data using MLPRegressor
from sklearn.neural_network import MLPRegressor

nn_scikit = MLPRegressor(hidden_layer_sizes=(16, 8),
                         activation='relu', solver='adam',
                         learning_rate_init=0.001,
                         random_state=42, max_iter=2000)

nn_scikit.fit(X_train, y_train)
predictions = nn_scikit.predict(X_test)
print(predictions)

print(np.mean((y_test - predictions) ** 2))

# yet another way to check data and run it against the prediction, tensorflow
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)

model = keras.Sequential([
    keras.layers.Dense(units=20, activation='relu'),
    keras.layers.Dense(units=8, activation='relu'),
    keras.layers.Dense(units=1)
])

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.02))

model.fit(X_train, y_train, epochs=300)

predictions = model.predict(X_test)[:, 0]
print(predictions)

print(np.mean((y_test - predictions) ** 2))

# This is using the keras model and tensorflow, specifically with dropout to keep overfitting low

model = keras.Sequential([
    keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    keras.layers.Dense(units=1)
])

# next requires data set. Pause. Use direct path because it is proper for me
mydata = pd.read_csv('C:/Users/Troy Cope/PycharmProjects/Assignment-01-CS4267-Troy-Cope/z20051201_20051210.csv',
                     index_col='Date')


# all these following definitions are for handling the data and organizing it
def add_original_feature(df, df_new):
    df_new['open'] = df['Open']
    df_new['open_1'] = df['Open'].shift(1)
    df_new['close_1'] = df['Close'].shift(1)
    df_new['high_1'] = df['High'].shift(1)
    df_new['low_1'] = df['Low'].shift(1)
    df_new['volume_1'] = df['Volume'].shift(1)


def add_avg_price(df, df_new):
    df_new['avg_price_5'] = df['Close'].rolling(5).mean().shift(1)
    df_new['avg_price_30'] = df['Close'].rolling(21).mean().shift(1)
    df_new['avg_price_365'] = df['Close'].rolling(252).mean().shift(1)
    df_new['ratio_avg_price_5_30'] = df_new['avg_price_5'] / df_new['avg_price_30']
    df_new['ratio_avg_price_5_365'] = df_new['avg_price_5'] / df_new['avg_price_365']
    df_new['ratio_avg_price_30_365'] = df_new['avg_price_30'] / df_new['avg_price_365']


def add_avg_volume(df, df_new):
    df_new['avg_volume_5'] = df['Volume'].rolling(5).mean().shift(1)
    df_new['avg_volume_30'] = df['Volume'].rolling(21).mean().shift(1)
    df_new['avg_volume_365'] = df['Volume'].rolling(252).mean().shift(1)
    df['ratio_avg_volume_5_30'] = df_new['avg_volume_5'] / df_new['avg_volume_30']
    df['ratio_avg_volume_5_365'] = df_new['avg_volume_5'] / df_new['avg_volume_365']
    df['ratio_avg_volume_30_365'] = df_new['avg_volume_30'] / df_new['avg_volume_365']


def add_std_price(df, df_new):
    df_new['std_price_5'] = df['Volume'].rolling(5).std().shift(1)
    df_new['std_price_30'] = df['Volume'].rolling(21).std().shift(1)
    df_new['std_price_365'] = df['Volume'].rolling(252).std().shift(1)
    df_new['ratio_std_volume_5_30'] = df_new['std_price_5'] / df_new['std_price_30']
    df_new['ratio_std_volume_5_365'] = df_new['std_price_5'] / df_new['std_price_365']
    df_new['ratio_std_volume_30_365'] = df_new['std_price_30'] / df_new['std_price_365']


def add_std_volume(df, df_new):
    df_new['std_volume_5'] = df['Volume'].rolling(5).std().shift(1)
    df_new['std_volume_30'] = df['Volume'].rolling(21).std().shift(1)
    df_new['std_volume_365'] = df['Volume'].rolling(252).std().shift(1)
    df_new['ratio_std_volume_5_30'] = df_new['std_volume_5'] / df_new['std_volume_30']
    df_new['ratio_std_volume_5_365'] = df_new['std_volume_5'] / df_new['std_volume_365']
    df_new['ratio_std_volume_30_365'] = df_new['std_volume_30'] / df_new['std_volume_365']


def add_return_feature(df, df_new):
    df_new['return_1'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).shift(1)
    df_new['return_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)).shift(1)
    df_new['return_30'] = ((df['Close'] - df['Close'].shift(21)) / df['Close'].shift(21)).shift(1)
    df_new['return_365'] = ((df['Close'] - df['Close'].shift(252)) / df['Close'].shift(252)).shift(1)
    df_new['moving_avg_5'] = df_new['return_1'].rolling(5).mean().shift(1)
    df_new['moving_avg_30'] = df_new['return_1'].rolling(21).mean().shift(1)
    df_new['moving_avg_365'] = df_new['return_1'].rolling(252).mean().shift(1)


# putting together all the features into a dataframe, so it can be compared
def generate_features(df):
    df_new = pd.DataFrame()
    add_original_feature(df, df_new)
    add_avg_price(df, df_new)
    add_avg_volume(df, df_new)
    add_std_price(df, df_new)
    add_std_volume(df, df_new)
    add_return_feature(df, df_new)
    df_new['close'] = df['Close']
    df_new = df_new.dropna(axis=0)
    return df_new


# I don't need this double call, but I went ahead and left it because
# I ran the program for 4 hours last time and don't want to do it again just to make sure it works still
data_raw = pd.read_csv('C:/Users/Troy Cope/PycharmProjects/Assignment-01-CS4267-Troy-Cope/z19880101_20191231.csv',
                       index_col='Date')
data = generate_features(data_raw)

print(data.round(decimals=3).head(5))

# setting up the training and test boundaries and data
start_train = '1988-01-01'
end_train = '2018-12-31'
start_test = '2019-01-01'
end_test = '2019-12-31'
data_train = data.loc[start_train:end_train]
X_train = data_train.drop('close', axis=1).values
y_train = data_train['close'].values
data_test = data.loc[start_test:end_test]
X_test = data_test.drop('close', axis=1).values
y_test = data_test['close'].values

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# scaling data to be more accurate and line up properly
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)

# line I added personally
predictions = model.predict(X_scaled_test)

# yet another trial of learning through tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(units=32, activation='relu'),
    Dense(units=1)
])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
model.fit(X_scaled_train, y_train, epochs=100, verbose=True)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# printing of values to show accuracy
print(f'MSE: {mean_squared_error(y_test, predictions):.3f}')
print(f'MAE: {mean_absolute_error(y_test, predictions):.3f}')
print(f'R^2: {r2_score(y_test, predictions):.3f}')

from tensorboard.plugins.hparams import api as hp

HP_HIDDEN = hp.HParam('hidden_size', hp.Discrete([64, 32, 16]))
HP_EPOCHS = hp.HParam('epochs', hp.Discrete([300, 1000]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(0.01, 0.4))

# defining a way to test the model, and sending back important values to be measured in tensorflow localhost
def train_test_model(hparams, logdir):
    model = Sequential([
        Dense(units=hparams[HP_HIDDEN],
              activation='relu'),
        Dense(units=1)
    ])
    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(hparams[HP_LEARNING_RATE]),
                  metrics=['mean_squared_error'])
    model.fit(X_scaled_train, y_train,
              validation_data=(X_scaled_test, y_test),
              epochs=hparams[HP_EPOCHS], verbose=False,
              callbacks=[
                  tf.keras.callbacks.TensorBoard(logdir),
                  hp.KerasCallback(logdir, hparams),
                  tf.keras.callbacks.EarlyStopping(
                      monitor='val_loss', min_delta=0,
                      patience=200, verbose=0,
                      mode='auto',
                  )
              ],
              )
    _, mse = model.evaluate(X_scaled_test, y_test)
    pred = model.predict(X_scaled_test)
    r2 = r2_score(y_test, pred)
    return mse, r2

# building the tensorflow localhost and shooting it data
def run(hparams, logdir):
    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams_config(
            hparams=[HP_HIDDEN, HP_EPOCHS,
                     HP_LEARNING_RATE],
            metrics=[hp.Metric('mean_squared_error', display_name='mse'),
                     hp.Metric('r2', display_name='r2')],
        )
        mse, r2 = train_test_model(hparams, logdir)
        tf.summary.scalar('mean_squared_error', mse, step=1)
        tf.summary.scalar('r2', r2, step=1)

# compiling the results for each session instance
session_num = 0
for hidden in HP_HIDDEN.domain.values:
    for epochs in HP_EPOCHS.domain.values:
        for learning_rate in tf.linspace(HP_LEARNING_RATE.domain.min_value,
                                         HP_LEARNING_RATE.domain.max_value,
                                         5):
            hparams = {
                HP_HIDDEN: hidden,
                HP_EPOCHS: epochs,
                HP_LEARNING_RATE:
                    float("%.2f" % float(learning_rate)),
            }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run(hparams, 'logs/hparams_tuning/' + run_name)
            session_num += 1

# These are my ideals, HS: 64, EP: 1000, LR: 0.3
model = Sequential([
    Dense(units=64, activation='relu'),
    Dense(units=1)
])

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.3))
model.fit(X_scaled_train, y_train, epochs=1000, verbose=False)
predictions = model.predict(X_scaled_test)[:, 0]

# plotting the data found
import matplotlib.pyplot as plt

plt.plot(data_test.index, y_test, c='k')
plt.plot(data_test.index, predictions, c='b')
plt.plot(data_test.index, predictions, c='r')
plt.plot(data_test.index, predictions, c='g')
plt.xticks(range(0, 252, 10), rotation=60)
plt.xlabel('Date')
plt.ylabel('Close price')
plt.legend(['Truth', 'Neural network prediciton'])
plt.show()
