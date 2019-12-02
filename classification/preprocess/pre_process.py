# coding=utf-8
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os
from sklearn import preprocessing


def preprocess_data(work_dir=''):
    train = pd.read_csv(work_dir + 'train_set.csv')
    jobs, maritals, educations, contacts = [], [], [], []
    for i in range(train.ID.size):
        if not jobs.__contains__(train.job.get(i)):
            jobs.append(train.job.get(i))
        if not maritals.__contains__(train.marital.get(i)):
            maritals.append(train.marital.get(i))
        if not educations.__contains__(train.education.get(i)):
            educations.append(train.education.get(i))
        if not contacts.__contains__(train.contact.get(i)):
            contacts.append(train.contact.get(i))

    months = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
              'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    X, Y = [], []
    for i in range(train.ID.size):
        features = np.zeros(34)
        features[0] = train.age.get(i)
        features[1 + jobs.index(train.job.get(i))] = 1
        features[13 + maritals.index(train.marital.get(i))] = 1
        features[16 + educations.index(train.education.get(i))] = 1
        features[20] = 1 if train.default.get(i) == 'yes' else 0
        features[21] = train.balance.get(i)
        features[22] = 1 if train.housing.get(i) == 'yes' else 0
        features[23] = 1 if train.loan.get(i) == 'yes' else 0
        features[24 + contacts.index(train.contact.get(i))] = 1
        features[27] = train.day.get(i)
        features[28] = months[train.month.get(i)]
        features[29] = train.duration.get(i)
        features[30] = train.campaign.get(i)
        features[31] = train.pdays.get(i)
        features[32] = train.previous.get(i)
        features[33] = 1 if train.default.get(i) == 'success' else 0

        X.append(features)
        Y.append(train.y.get(i))

    x = np.array(X)
    y = np.array(Y)
    x, y = build_balance_data(x, y)
    # 归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    np.savez(work_dir + "train_set_preprocess", x=x, y=y)

    pca = PCA(n_components=28)
    x_pca = pca.fit_transform(x)
    # 归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    x_pca = min_max_scaler.fit_transform(x_pca)

    np.savez(work_dir + "train_set_pca", x=x_pca, y=y)
    return x, y, x_pca, y


def build_balance_data(x, y):
    data_x = []
    data_y = []
    count1 = 0
    count0 = 0
    for i in range(y.shape[0]):
        if y[i] == 0:
            data_x.append(x[i])
            data_y.append(0)
            count0 += 1

    while count1 < count0:
        for i in range(y.shape[0]):
            if y[i] == 1:
                data_x.append(x[i])
                data_y.append(1)
                count1 += 1
                if count1 >= count0:
                    break

    data_x = np.array(data_x)
    data_y = np.array(data_y)
    per = np.random.permutation(data_x.shape[0])

    data_x = data_x[per]
    data_y = data_y[per]

    return data_x, data_y


def load_train_data(work_dir=''):
    if os.path.exists(work_dir + "train_set_preprocess" + ".npz"):
        zip_file = np.load(work_dir + "train_set_preprocess" + ".npz")
        return zip_file['x'], zip_file['y']

    print("Can not find dump data, build from csv, please wait...")
    return preprocess_data(work_dir)[:2]


def load_train_data_pca(work_dir=''):
    if os.path.exists(work_dir + "train_set_pca" + ".npz"):
        zip_file = np.load(work_dir + "train_set_pca" + ".npz")
        return zip_file['x'], zip_file['y']

    print("Can not find dump data, build from csv, please wait...")
    return preprocess_data(work_dir)[2:]


if __name__ == '__main__':
    x, y, x_pca, y = preprocess_data()
    print(x.shape, x_pca.shape)
    # x, y = load_train_data_pca()
    # x1 = load_train_data()[0]
    # print(x.shape, x1.shape)
