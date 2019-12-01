# coding=utf-8
import pandas as pd
import numpy as np

train = pd.read_csv('train_set.csv')
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
per = np.random.permutation(x.shape[0])
x = x[per]
y = y[per]

print(x.shape, y.shape)
