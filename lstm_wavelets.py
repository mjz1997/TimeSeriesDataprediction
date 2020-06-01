# import pywt
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.style
from PyEMD import EMD
from torch import nn
import torch
import pickle
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler


class LSTM_torch(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, predict_steps):
        super(LSTM_torch, self).__init__()

        self.predict_steps = predict_steps
        self.lstm1 = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size1,
                             num_layers=2,
                             batch_first=True,
                             dropout=0.3)
        self.dropout1 = nn.Dropout(p=0.3)
        self.lstm2 = nn.LSTM(input_size=hidden_size1,
                             hidden_size=output_size,
                             num_layers=1,
                             batch_first=True)
        # self.dropout2 = nn.Dropout(p=0.2)
        # self.lstm3 = nn.LSTM(input_size=hidden_size2,
        #                      hidden_size=output_size,
        #                      num_layers=1,
        #                      batch_first=True)

    def forward(self, input):
        # h0, c0 = torch.randn(size=())
        lstm1_out, (hn_1, cn_1) = self.lstm1(input)
        # lstm1_out = self.dropout1(lstm1_out)
        lstm2_out, (hn_2, cn_2) = self.lstm2(lstm1_out)
        # lstm2_out = self.dropout2(lstm2_out)
        # lstm3_out, (hn_3, cn_3) = self.lstm3(lstm2_out)
        return lstm2_out[:, -self.predict_steps:, :]


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


def plot_result1(date_ini, ts_ini, date_oup_train, ts_oup_train_pre, date_oup_test, ts_oup_test_pre,
                train_r2, test_r2, model_name, time_steps):
    # plot
    # sns.set()
    plt.style.use('ggplot')
    # plt.style.use('seaborn-pastel')
    figures_path = os.curdir + '/figures'
    mkdir(figures_path)
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    label1 = 'training set stress prediction  ' + ' R2: ' + str(train_r2)
    ax1.plot(date_ini[-10*ts_oup_test_pre.shape[0]:], ts_ini[-10*ts_oup_test_pre.shape[0]:], color='black',
             linestyle='--', linewidth=1, label='original stress data')
    ax1.plot(date_oup_train[-8*ts_oup_test_pre.shape[0]:], ts_oup_train_pre[-8*ts_oup_test_pre.shape[0]:],
              linewidth=1.5, label=label1)
    label2 = 'test set stress prediction  ' + ' R2: ' + str(test_r2)
    ax1.plot(date_oup_test, ts_oup_test_pre, linewidth=1.5, label=label2)
    # ax1.set_xlabel('shear strain, $γ$', fontsize=15, labelpad=5)
    ax1.set_xlabel('Experimental run time (s)', fontsize=15, labelpad=5)
    ax1.set_ylabel('shear stress, $τ$ (MPa)', fontsize=15, labelpad=5)
    ax1.tick_params(axis='both', labelsize=15)
    plt.title(model_name, fontsize=20)
    fig.legend(loc='center', fontsize=10)
    # plt.grid(axis='both', color='grey', linestyle='--', lw=0.5, alpha=0.5)
    fig_name = figures_path + '/' + model_name + '   time_steps=' + str(time_steps) + '.png'
    plt.savefig(fig_name, dpi=600, bbox_inches='tight')
    plt.show()


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


def lstm_prediction_torch(date_ini, ts_ini, time_steps, predict_steps, groups, minmax_scaler, test_scale):

    # initiate the model and the dataset
    n_features = ts_ini.shape[0]//groups
    model = LSTM_torch(n_features, 10*n_features, 5*n_features, n_features, predict_steps)
    ts_inp, ts_oup, date_oup, initial_ts = data_for_lstm(date_ini, ts_ini, time_steps, predict_steps, groups)

    # turn the shape of output into the shape(batch_size, seq_len, features), which the torch needs.
    ts_oup = ts_oup.reshape((-1, predict_steps, n_features))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # divide the dataset into training set and test set with the train_size
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

    # prepare the dataset for lstm based on torch.units.data (TensorDataset and DataLoader)
    Tensor_ts_inp_train = torch.from_numpy(ts_inp_train.astype(np.float32))
    Tensor_ts_oup_train = torch.from_numpy(ts_oup_train.astype(np.float32))
    train_dataset = TensorDataset(Tensor_ts_inp_train, Tensor_ts_oup_train)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # train the model on the training set
    if groups > 5000:
        epochs = 200
        batch_size = int(groups / 50) # 50
    else:
        epochs = 200
        batch_size = int(groups / 25) # 25
    lr = 1e-3

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)     # initiate the optimizer
    loss_fn = nn.MSELoss()                                  # initiate the loss function

    for epoch in range(epochs):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x.requires_grad_(True), batch_y.requires_grad_(True)
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            print('Epoch:', epoch+1, 'Training...\n', 'Step:', step+1, 'Loss:', loss)

    Tensor_ts_oup_train_pre = model(Tensor_ts_inp_train)
    ts_oup_train_pre = Tensor_ts_oup_train_pre.detach().numpy()
    ts_oup_train_pre1 = ts_oup_train_pre[:, 0, :].reshape((-1, n_features))
    train_r2 = round(r2_score(initial_ts_train, ts_oup_train_pre1), 5)
    print(train_r2)
    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # prediction
    ts_inp_tol = ts_inp_test[0]
    ts_oup_test_pre = []
    a = 0
    while a < initial_ts_test.shape[0]:
        ts_inp_tep = np.array(ts_inp_tol[-time_steps:]).reshape((1, time_steps, n_features))
        Tensor_inp_tep = torch.from_numpy(ts_inp_tep.astype(np.float32))
        Tensor_ts_oup_tep = model(Tensor_inp_tep)
        ts_oup_tep = Tensor_ts_oup_tep.detach().numpy()  # shape: (1, predict_steps, n_features)
        ts_oup_test_pre.append(ts_oup_tep[0])
        ts_oup_tep = ts_oup_tep.reshape((-1, n_features))
        ts_inp_tol = np.vstack((ts_inp_tol, ts_oup_tep))
        a = ts_inp_tol.shape[0]-time_steps
    ts_oup_test_pre = np.asarray(ts_oup_test_pre).reshape((-1, n_features))
    ts_oup_test_pre = ts_oup_test_pre[:initial_ts_test.shape[0]]
    # print(ts_inp_tol)
    test_r2 = round(r2_score(initial_ts_test, ts_oup_test_pre), 5)
    #
    # # reshape the ts_oup into a column for plot
    ts_oup_train_pre1 = ts_oup_train_pre1.reshape((-1, 1))
    ts_oup_test_pre = ts_oup_test_pre.reshape((-1, 1))
    #
    ts_oup_train_pre1 = minmax_scaler.inverse_transform(ts_oup_train_pre1)
    ts_oup_test_pre = minmax_scaler.inverse_transform(ts_oup_test_pre)
    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return ts_oup_train_pre1, ts_oup_test_pre, date_oup_train, date_oup_test, train_r2, test_r2


def empirical_mode_decomposition(date_ini, ts_ini):
    """
    :param ts_ini: the raw data to be decomposed
    :return: the decomposed time series
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//the wavelet tranform //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # A2, D2, D1 = pywt.wavedec(stress_ini.ravel(), 'db5', level=2)
    # print(stress_ini.shape)
    # print(D2.shape, D1.shape)
    # cow = [A2, D2, D1]
    # raw_data = pywt.waverec(cow, 'db5')
    # plt.style.use('seaborn-pastel')
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111)
    # ax.plot(stress_ini)
    # ax.plot(A2)
    # ax.plot(D2)
    # ax.plot(D1)
    # ax.plot(raw_data+0.1)
    # plt.show()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//the empirical mode decomposition //~~~~~~~~~~~~~~~~~~~~~
    emd = EMD()
    emd.emd(ts_ini.ravel())
    imfs, res = emd.get_imfs_and_residue()
    # Plot results
    # N = imfs.shape[0] + 2
    # plt.style.use('seaborn-pastel')
    # plt.figure(figsize=(15, 20))
    # plt.subplot(N, 1, 1)
    # plt.plot(date_ini, ts_ini, 'r')
    # plt.title("Raw data")
    # plt.xlabel('shear stress, ' + r'$\tau$(MPa)')
    #
    # for n, imf in enumerate(imfs):
    #     plt.subplot(N, 1, n + 2)
    #     plt.plot(date_ini, imf, 'g')
    #     plt.title("IMF " + str(n + 1))
    #     plt.xlabel("Time [0.1s]")
    #
    # plt.subplot(N, 1, N)
    # plt.plot(date_ini, res, 'b')
    # plt.title("residue")
    # plt.xlabel('shear stress, ' + r'$\tau$(MPa)')
    # plt.tight_layout()
    # plt.savefig('simple_example', dpi=600, bbox_inches='tight')
    # plt.show()
    return imfs, res
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def main():
    # ~~~~~~~~~~~// Input the experimental data  from the following reference // ~~~~~~~~~~~~~~
    # reference: Rouet-Leduc, B(2018). Estimating Fault Friction From Seismic Signals in the Laboratory.
    # Geophysical Research Letters, 45(3), 1321–1329. https://doi.org/10.1002/2017GL076708
    # strain_ini = pd.read_excel(os.path.curdir + '/Time Series Data/p4677_t2000_2500.xlsx', usecols=[0])
    # stress_ini = pd.read_excel(os.path.curdir + '/Time Series Data/p4677_t2000_2500.xlsx', usecols=[1])
    # strain_ini, stress_ini = np.array(strain_ini)[::100], np.array(stress_ini)[::100]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~// Input the experimental data  from the following reference // ~~~~~~~~~~~~~~
    # reference: DEM experimental data from magang
    strain_ini = pd.read_excel(os.path.curdir + '/strain_stress_500000_0.1~1.9d50.xls', usecols=[1])
    stress_ini = pd.read_excel(os.path.curdir + '/strain_stress_500000_0.1~1.9d50.xls', usecols=[3])
    strain_ini, stress_ini = np.array(strain_ini)[::100], np.array(stress_ini)[::100]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    imfs, res = empirical_mode_decomposition(strain_ini, stress_ini)
    time_steps = 20
    predict_steps = 1
    groups = 5000
    test_scale = 0.02
    N = imfs.shape[0]

    tol_train_results = []
    tol_test_results = []
    for n, imf in enumerate(imfs):
        imf = np.array(imf).reshape((-1, 1))
        minmax_scaler = MinMaxScaler().fit(imf)
        stress_transformed = minmax_scaler.transform(imf)
        ts_oup_train_pre1, ts_oup_test_pre, date_oup_train, date_oup_test, train_r2, test_r2 = \
            lstm_prediction_torch(strain_ini, stress_transformed, time_steps, predict_steps, groups,
                                  minmax_scaler, test_scale)
        tol_train_results.append(ts_oup_train_pre1.ravel())
        tol_test_results.append(ts_oup_test_pre.ravel())
        model_name = 'LSTM ' + str(n)
        plot_result1(strain_ini, imf, date_oup_train, ts_oup_train_pre1, date_oup_test,
                         ts_oup_test_pre, train_r2, test_r2, model_name, time_steps)

    tol_train_results = np.array(tol_train_results).reshape((N, -1)).T
    tol_test_results = np.array(tol_test_results).reshape((N, -1)).T

    # fw = open('result.dump', 'wb')
    # pickle.dump(tol_train_results, fw)
    # pickle.dump(tol_test_results. fw)
    # fw.close()

    model_name = 'LSTM '
    add_train = tol_train_results.sum(axis=1)
    add_test = tol_test_results.sum(axis=1)
    add_train = add_train.reshape((-1, 1))
    add_test = add_test.reshape((-1, 1))
    a, b = add_train.shape[0], add_test.shape[0]
    train_r2 = r2_score(stress_ini[:a], add_train)
    test_r2 = r2_score(stress_ini[-b:], add_test)
    plot_result1(strain_ini, stress_ini, date_oup_train, add_train, date_oup_test, add_test, train_r2,
                 test_r2, model_name, time_steps)

    print([stress_ini[:a], add_train])


if __name__ == '__main__':
    main()