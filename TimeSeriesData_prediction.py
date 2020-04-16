# !/usr/bin/env python
# -*- coding: UTF-8 -*-
# script to predict time series data  --- by meijz
# References:
# 1.Long Short-Term Memory layer - Hochreiter 1997.
# 2.https://keras.io/layers/recurrent/
# 3.Learning to forget: Continual prediction with LSTM
# 4.Supervised sequence labeling with recurrent neural networks
# 5.A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
# 6.https://www.toutiao.com/a6815029562854867467/

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import openpyxl
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb


def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("/")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        print(path + ' 创建成功')
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False


def plot_autocorrelation(ts_ini):
    # calculate the acf and pacf of time series data
    # acfs = acf(ts_ini, nlags=2000)
    # pacfs = pacf(ts_ini, nlags=50)
    plot_acf(ts_ini, lags=2000)
    plot_pacf(ts_ini, lags=50)
    # acovfs = acovf()
    # plot
    # sns.set()
    # plt.figure(figsize=(6, 8))
    # plt.subplot(2, 1, 1)
    # plt.plot(acfs, color='red', linestyle='--', linewidth=1.5, label='acf')
    # plt.xlabel('nlags', fontsize=15, labelpad=5)
    # plt.ylabel('acf', fontsize=15, labelpad=5)
    # plt.tick_params(axis='both', labelsize=15)
    # plt.legend(loc='center', fontsize=15)
    # plt.grid(axis='both', color='grey', linestyle='--', lw=0.5, alpha=0.5)
    # plt.subplot(2, 1, 2)
    # plt.plot(pacfs, color='b', linestyle=':', linewidth=1.5, label='pacf')
    # plt.xlabel('nlags', fontsize=15, labelpad=5)
    # plt.ylabel('pacf', fontsize=15, labelpad=5)
    # plt.tick_params(axis='both', labelsize=15)
    # plt.legend(loc='center', fontsize=15)
    # plt.grid(axis='both', color='grey', linestyle='--', lw=0.5, alpha=0.5)
    # plt.savefig('autocorrelation.png', dpi=600, bbox_inches='tight')
    # plt.show()


def calc_features(input):
    features = []
    for i in range(input.shape[0]):
        para_first = input[i, 0]
        para_max = np.max(input[i])
        para_min = np.min(input[i])
        para_aver = np.average(input[i])
        para_norm = np.linalg.norm(input[i], ord=2)

        features.append([para_first, para_max, para_min, para_aver, para_norm])
    features = np.array(features).reshape((-1, 5))
    return features


def plot_result1(date_ini, ts_ini, date_oup_train, ts_oup_train_pre, date_oup_test, ts_oup_test_pre,
                train_r2, test_r2, model_name):
    # plot
    # sns.set()
    figures_path = os.curdir + '/figures'
    mkdir(figures_path)
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    label1 = 'training set stress prediction  ' + ' R2: ' + str(train_r2)
    ax1.plot(date_ini[-4*ts_oup_test_pre.shape[0]:], ts_ini[-4*ts_oup_test_pre.shape[0]:], color='black',
             linestyle='--', linewidth=1, label='original stress data')
    ax1.plot(date_oup_train[-ts_oup_test_pre.shape[0]:], ts_oup_train_pre[-ts_oup_test_pre.shape[0]:],
             color='r', linewidth=1.5, label=label1)
    label2 = 'test set stress prediction  ' + ' R2: ' + str(test_r2)
    ax1.plot(date_oup_test, ts_oup_test_pre, color='b', linewidth=1.5, label=label2)
    ax1.set_xlabel('shear strain, $γ$', fontsize=15, labelpad=5)
    ax1.set_ylabel('shear stress, $τ$ (MPa)', fontsize=15, labelpad=5)
    ax1.tick_params(axis='both', labelsize=15)
    plt.title(model_name, fontsize=20)
    fig.legend(loc='center', fontsize=10)
    plt.grid(axis='both', color='grey', linestyle='--', lw=0.5, alpha=0.5)
    fig_name = figures_path + '/' + model_name + '.png'
    plt.savefig(fig_name, dpi=600, bbox_inches='tight')
    plt.show()


def plot_result2(date_ini, ts_ini, date_oup_test, ts_oup_test_pre, test_r2, model_name):
    # plot
    # sns.set()
    figures_path = os.curdir + '/figures'
    mkdir(figures_path)
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(date_ini[-4*ts_oup_test_pre.shape[0]:], ts_ini[-4*ts_oup_test_pre.shape[0]:], color='black',
             linestyle='--', linewidth=1, label='original stress data')
    label2 = 'test set stress prediction  ' + ' R2: ' + str(test_r2)
    ax1.plot(date_oup_test, ts_oup_test_pre, color='b', linewidth=1.5, label=label2)
    ax1.set_xlabel('shear strain, $γ$', fontsize=15, labelpad=5)
    ax1.set_ylabel('shear stress, $τ$ (MPa)', fontsize=15, labelpad=5)
    ax1.tick_params(axis='both', labelsize=15)
    plt.title(model_name, fontsize=20)
    fig.legend(loc='center', fontsize=10)
    plt.grid(axis='both', color='grey', linestyle='--', lw=0.5, alpha=0.5)
    # fig_name = figures_path + '/Macroscopic responses-groups' + str(groups) + '-timesteps' + str(
    #     time_steps) + '-presteps' + str(predict_steps) + '.png'
    fig_name = figures_path + '/' + model_name + '.png'
    plt.savefig(fig_name, dpi=600, bbox_inches='tight')
    plt.show()


def model_SimpleExpSmoothing(time_series, span):
    """
    reference: Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles and practice. OTexts, 2014.
    """
    alpha = 2/(span+1)
    model = SimpleExpSmoothing(time_series).fit(smoothing_level=alpha, optimized=False)
    return model


def model_SecondExpSmoothing(time_series, span):
    """
    SecondExpSmoothing adds the trend compared to SimpleExpSmoothing
    """
    model = ExponentialSmoothing(time_series, trend='add', seasonal_periods=span).fit()
    return model


def model_ThirdExpSmoothing(time_series, span):
    """
    ThirdExpSmoothing adds the trend and the seasonal items compared to SimpleExpSmoothing
    """
    model = ExponentialSmoothing(time_series, trend='add', seasonal='add', seasonal_periods=span).fit()
    return model


def model_Arima(time_series, max_p, max_q, model_name='ARIMA'):
    """
    Initiate the arima model based on pmdarima, which can find the best (p,d,q) given the time series.
    model_name: 'ARIMA'/'SARIMA'/'SARIMAX'
    model SARIMAX can only consider extra feature(little use)
    """
    if model_name == 'ARIMA':
        model = auto_arima(time_series, max_p=max_p, max_q=max_q, seasonal=False)
    elif model_name == 'SARIMA':
        model = auto_arima(time_series, max_p=max_p, max_q=max_q)
    elif model_name == 'SARIMAX':
        print('choose another one')
    return model


def model_lightgbm():
    model = lgb.LGBMRegressor()
    return model


def model_randomforests():
    model = RandomForestRegressor()
    return model


def model_lstm(units, input_shape):
    """
    initiate a lstm model based on keras
    units:[layer1_nodes, layer2_nodes,...], the last layer should be a dense layer for output with one time_step
    input_shape: arrays like; [time_steps, n_features]
    """
    model = Sequential()
    for i in range(len(units)-1):
        if i == 0:
            model.add(LSTM(units=units[i], input_shape=input_shape, return_sequences=True))
            model.add(Dropout(0.1))
            continue
        elif i == len(units)-2:
            model.add(LSTM(units=units[i], return_sequences=False))
            model.add(Dropout(0.1))
            continue
        model.add(LSTM(units=units[i], return_sequences=True))
        model.add(Dropout(0.1))
    model.add(Dense(units[-1], activation='sigmoid'))
    model.compile(optimizer='Adam', loss='mean_squared_error')
    model.summary()
    return model


def data_for_classicalmodels(date_ini, ts_ini, test_scale):
    """
    :param ts_ini: initial time series data
    :param test_scale: range(0, 1)
    :return: divided dataset for classical models
    """
    train_num = int((1-test_scale)*ts_ini.shape[0])
    date_train, ts_train = date_ini[:train_num], ts_ini[:train_num]
    date_test, ts_test = date_ini[train_num:], ts_ini[train_num:]
    return date_train, ts_train, date_test, ts_test


def data_for_MLmodels(date_ini, ts_ini, time_steps, predict_steps):

    sample_tol = ts_ini.shape[0]
    a = time_steps
    b = 0
    initial_ts = ts_ini[time_steps:]
    ts_inp = np.zeros(shape=(groups - time_steps - predict_steps + 1, time_steps))
    ts_oup = np.zeros(shape=(groups - time_steps - predict_steps + 1, predict_steps))
    # the ts_inp's shape is （n_samples, time_steps, n_features）
    while a < groups - predict_steps + 1:
        ts_inp[b] = ts_ini[b:(b + time_steps)].reshape((1, -1))[0]
        ts_oup[b] = ts_ini[b + time_steps:b + time_steps + predict_steps].reshape((1, -1))[0]
        a += 1
        b += 1
    ts_oup = ts_oup.reshape((groups - time_steps - predict_steps + 1, predict_steps))
    date_oup = np.asarray(date_ini).reshape((-1, 1))
    date_oup = date_oup[time_steps:, :]
    return ts_inp, ts_oup, date_oup, initial_ts


def MLmodels_prediction(model, date_ini, ts_ini, time_steps, predict_steps, minmax_scaler, test_scale):

    # initiate the model and the dataset
    ts_inp, ts_oup, date_oup, initial_ts = data_for_MLmodels(date_ini, ts_ini, time_steps, predict_steps)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # divide the dataset into training set and test set
    samples_tol = ts_inp.shape[0]
    train_num = int((1 - test_scale) * samples_tol)
    ts_inp_train, ts_oup_train = ts_inp[:train_num], ts_oup[:train_num]
    ts_inp_test, ts_oup_test = ts_inp[train_num:], ts_oup[train_num:]
    date_oup_train = date_oup[:train_num]
    date_oup_test = date_oup[train_num:]
    initial_ts_train = initial_ts[:train_num]
    initial_ts_test = initial_ts[train_num:]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # train the model based on the training set
    features = calc_features(ts_inp_train)
    model.fit(features, ts_oup_train)
    ts_oup_train_pre = model.predict(features)
    ts_oup_train_pre = np.array(ts_oup_train_pre).reshape((-1, predict_steps))
    print(ts_oup_train_pre)
    ts_oup_train_pre1 = ts_oup_train_pre[:, 0]
    train_r2 = round(r2_score(initial_ts_train, ts_oup_train_pre1), 5)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # prediction
    ts_inp_tol = ts_inp_test[0].reshape((-1, 1))
    ts_oup_test_pre = []
    a = 0
    while a < initial_ts_test.shape[0]:
        ts_inp_tep = np.array(ts_inp_tol[-time_steps:]).reshape((1, -1))
        feature_tep = calc_features(ts_inp_tep)
        ts_oup_tep = model.predict(feature_tep)
        ts_oup_test_pre.append(ts_oup_tep[0])
        ts_oup_tep = ts_oup_tep.reshape((-1, 1))
        ts_inp_tol = np.vstack((ts_inp_tol, ts_oup_tep))
        a = ts_inp_tol.shape[0] - time_steps
    ts_oup_test_pre = np.asarray(ts_oup_test_pre).reshape((-1, 1))
    ts_oup_test_pre = ts_oup_test_pre[:initial_ts_test.shape[0]]
    print(ts_inp_tol)
    test_r2 = round(r2_score(initial_ts_test, ts_oup_test_pre), 5)

    # reshape the ts_oup into a column for plot
    ts_oup_train_pre1 = ts_oup_train_pre1.reshape((-1, 1))
    ts_oup_test_pre = ts_oup_test_pre.reshape((-1, 1))

    ts_oup_train_pre1 = minmax_scaler.inverse_transform(ts_oup_train_pre1)
    ts_oup_test_pre = minmax_scaler.inverse_transform(ts_oup_test_pre)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return ts_oup_train_pre1, ts_oup_test_pre, date_oup_train, date_oup_test, train_r2, test_r2


def data_for_lstm(date_ini, ts_ini, time_steps, predict_steps, groups):
    """
    turn input and output into data with the given time_steps for the supervised tasks in lstm
    :param time_steps:
    :param predict_steps:
    :param groups:
    :return:
    """
    sample_tol = ts_ini.shape[0]
    ts_tep = ts_ini.reshape((groups, -1))
    n_features = sample_tol // groups
    a = time_steps
    b = 0
    #
    initial_ts = ts_tep[time_steps:]
    ts_inp = np.zeros(shape=(groups-time_steps-predict_steps+1, time_steps, n_features))
    ts_oup = np.zeros(shape=(groups-time_steps-predict_steps+1, predict_steps, n_features))
    # the ts_inp's shape is （n_samples, time_steps, n_features）
    while a < groups-predict_steps+1:
        ts_inp[b] = ts_tep[b:(b+time_steps), :]
        ts_oup[b] = ts_tep[b+time_steps:b+time_steps+predict_steps]
        a += 1
        b += 1
    ts_oup = ts_oup.reshape((groups-time_steps-predict_steps+1, predict_steps*n_features))
    date_oup = np.asarray(date_ini).reshape((-1, 1))
    date_oup = date_oup[time_steps*n_features:, :]
    return ts_inp, ts_oup, date_oup, initial_ts


def lstm_prediction(date_ini, ts_ini, time_steps, predict_steps, groups, minmax_scaler, test_scale):

    # initiate the model and the dataset
    n_features = ts_ini.shape[0]//groups
    units = [10*n_features, 5*n_features, predict_steps*n_features]
    input_shape = (time_steps, n_features)
    model = model_lstm(units, input_shape)
    ts_inp, ts_oup, date_oup, initial_ts = data_for_lstm(date_ini, ts_ini, time_steps, predict_steps, groups)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # divide the dataset into training set and test set with the train_size of 0.8
    samples_tol = ts_inp.shape[0]
    test_scale = test_scale/n_features
    train_num = int((1-test_scale) * samples_tol)
    ts_inp_train, ts_oup_train = ts_inp[:train_num], ts_oup[:train_num]
    ts_inp_test, ts_oup_test = ts_inp[train_num:], ts_oup[train_num:]
    date_oup_train = date_oup[:train_num*n_features]
    date_oup_test = date_oup[train_num*n_features:]
    initial_ts_train = initial_ts[:train_num]
    initial_ts_test = initial_ts[train_num:]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # train the model based on the training set
    if groups > 5000:
        epochs = 200
        batch_size = int(groups/100)
    else:
        epochs = 200
        batch_size = int(groups/25)
    model.fit(ts_inp_train, ts_oup_train, epochs=epochs, batch_size=batch_size)
    ts_oup_train_pre = model.predict(ts_inp_train)
    ts_oup_train_pre1 = ts_oup_train_pre[:, :n_features]
    train_r2 = round(r2_score(initial_ts_train, ts_oup_train_pre1), 5)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # prediction
    ts_inp_tol = ts_inp_test[0]
    ts_oup_test_pre = []
    a = 0
    while a < initial_ts_test.shape[0]:
        ts_inp_tep = np.array(ts_inp_tol[-time_steps:]).reshape((1, time_steps, n_features))
        ts_oup_tep = model.predict(ts_inp_tep)
        ts_oup_test_pre.append(ts_oup_tep[0])
        ts_oup_tep = ts_oup_tep.reshape((-1, n_features))
        ts_inp_tol = np.vstack((ts_inp_tol, ts_oup_tep))
        a = ts_inp_tol.shape[0]-time_steps
    ts_oup_test_pre = np.asarray(ts_oup_test_pre).reshape((-1, n_features))
    ts_oup_test_pre = ts_oup_test_pre[:initial_ts_test.shape[0]]
    print(ts_inp_tol)
    test_r2 = round(r2_score(initial_ts_test, ts_oup_test_pre), 5)

    # reshape the ts_oup into a column for plot
    ts_oup_train_pre1 = ts_oup_train_pre1.reshape((-1, 1))
    ts_oup_test_pre = ts_oup_test_pre.reshape((-1, 1))

    ts_oup_train_pre1 = minmax_scaler.inverse_transform(ts_oup_train_pre1)
    ts_oup_test_pre = minmax_scaler.inverse_transform(ts_oup_test_pre)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return ts_oup_train_pre1, ts_oup_test_pre, date_oup_train, date_oup_test, train_r2, test_r2


def TimeSeriesPrediction(model_name, time_steps, predict_steps, groups, test_scale, span):
    """
    :param model_name: 'SimpleExpSmoothing'/'SecondExponentialSmoothing'/'ThirdExponentialSmoothing'/
        'ARIMA'/'SARIMA'/'SARIMAX'/'lstm'
    """
    # ~~~~~~~~~~~// The stress-strain data of simple shear simulation// ~~~~~~~~~~~~~~~~~~~~~~
    strain_ini = pd.read_excel(os.path.curdir + '/Time Series Data/strain_stress_5000.xls', usecols=[1])
    stress_ini = pd.read_excel(os.path.curdir + '/Time Series Data/strain_stress_5000.xls', usecols=[3])
    strain_ini, stress_ini= np.array(strain_ini)[1000:], np.array(stress_ini)[1000:]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~// The sale data of 'M5-forecast-accuracy' from kaggle // ~~~~~~~~~~~~~~~~~~~~~
    # strain_ini = pd.read_excel(os.path.curdir + '/Time Series Data/m5_DATA.xlsx', usecols=[0])
    # stress_ini = pd.read_excel(os.path.curdir + '/Time Series Data/m5_DATA.xlsx', usecols=[1])
    # strain_ini, stress_ini= np.array(strain_ini), np.array(stress_ini)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # fig1 = plot_acf(stress_ini, lags=500)
    # fig2 = plot_pacf(stress_ini, lags=50)
    # fig1.show()
    # fig2.show()

    minmax_scaler = MinMaxScaler().fit(stress_ini)
    stress_transformed = minmax_scaler.transform(stress_ini)

    if model_name == 'lstm':
        ts_oup_train_pre1, ts_oup_test_pre, date_oup_train, date_oup_test, train_r2, test_r2 = lstm_prediction(
            strain_ini, stress_transformed, time_steps, predict_steps, groups, minmax_scaler, test_scale)
        plot_result1(strain_ini, stress_ini, date_oup_train, ts_oup_train_pre1, date_oup_test, ts_oup_test_pre,
                     train_r2, test_r2, model_name)

    elif model_name == 'SimpleExpSmoothing':
        date_train, ts_train, date_test, ts_test = data_for_classicalmodels(strain_ini, stress_ini,
                                                                            test_scale=test_scale)
        model = model_SimpleExpSmoothing(ts_train, span)
        ts_test_pre = model.forecast(ts_test.shape[0])
        test_r2 = r2_score(ts_test, ts_test_pre)
        plot_result2(strain_ini, stress_ini, date_test, ts_test_pre, test_r2, model_name)

    elif model_name == 'SecondExpSmoothing':
        date_train, ts_train, date_test, ts_test = data_for_classicalmodels(strain_ini, stress_ini,
                                                                            test_scale=test_scale)
        model = model_SecondExpSmoothing(ts_train, span)
        ts_test_pre = model.forecast(ts_test.shape[0])
        test_r2 = r2_score(ts_test, ts_test_pre)
        plot_result2(strain_ini, stress_ini, date_test, ts_test_pre, test_r2, model_name)

    elif model_name == 'ThirdExpSmoothing':
        date_train, ts_train, date_test, ts_test = data_for_classicalmodels(strain_ini, stress_ini,
                                                                            test_scale=test_scale)
        model = model_ThirdExpSmoothing(ts_train, span)
        ts_test_pre = model.forecast(ts_test.shape[0])
        test_r2 = r2_score(ts_test, ts_test_pre)
        plot_result2(strain_ini, stress_ini, date_test, ts_test_pre, test_r2, model_name)

    elif model_name == 'ARIMA':
        date_train, ts_train, date_test, ts_test = data_for_classicalmodels(strain_ini, stress_ini,
                                                                            test_scale=test_scale)
        model = model_Arima(ts_train, 100, 20)
        model.fit(ts_train)
        ts_test_pre = model.predict(ts_test.shape[0])
        test_r2 = r2_score(ts_test, ts_test_pre)
        plot_result2(strain_ini, stress_ini, date_test, ts_test_pre, test_r2, model_name)

    elif model_name == 'SARIMA':
        date_train, ts_train, date_test, ts_test = data_for_classicalmodels(strain_ini, stress_ini,
                                                                            test_scale=test_scale)
        model = model_Arima(ts_train, 100, 20, 'SARIMA')
        model.fit(ts_train)
        ts_test_pre = model.predict(ts_test.shape[0])
        test_r2 = r2_score(ts_test, ts_test_pre)
        plot_result2(strain_ini, stress_ini, date_test, ts_test_pre, test_r2, model_name)

    elif model_name == 'lightgbm':
        model = model_lightgbm()
        ts_oup_train_pre1, ts_oup_test_pre, date_oup_train, date_oup_test, train_r2, test_r2 = MLmodels_prediction(
            model,  strain_ini, stress_transformed, time_steps, predict_steps, minmax_scaler, test_scale)
        plot_result1(strain_ini, stress_ini, date_oup_train, ts_oup_train_pre1, date_oup_test, ts_oup_test_pre,
                     train_r2, test_r2, model_name)

    elif model_name == 'randomforests':
        model = model_randomforests()
        ts_oup_train_pre1, ts_oup_test_pre, date_oup_train, date_oup_test, train_r2, test_r2 = MLmodels_prediction(
            model, strain_ini, stress_transformed, time_steps, predict_steps, minmax_scaler, test_scale)
        plot_result1(strain_ini, stress_ini, date_oup_train, ts_oup_train_pre1, date_oup_test, ts_oup_test_pre,
                     train_r2, test_r2, model_name)

    # elif model_name


if __name__ == '__main__':

    # for groups in [5000, 1000]:
    #     for time_steps in range(400, 1000, 100):
    #         for predict_steps in range(5, 55, 5):
    #             # time_steps = 40
    #             # predict_steps = 5
    #             strain_interval_num = 5000
    #             # groups = 5000
    #             n_features = strain_interval_num // groups
    #             units = [n_features*10, n_features*10, n_features*predict_steps]
    #             stress_prediction(units, n_features, strain_interval_num, time_steps, predict_steps, groups)

    time_steps = 100
    predict_steps = 1
    groups = 4000
    test_scale = 0.025
    span = 100

    for model_name in ['SimpleExpSmoothing', 'SecondExpSmoothing', 'ThirdExpSmoothing', 'ARIMA',
                       'SARIMA', 'lightgbm', 'randomforests']:
        TimeSeriesPrediction(model_name, time_steps, predict_steps, groups, test_scale, span)