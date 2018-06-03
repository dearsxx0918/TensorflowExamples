import os
import time

import pandas as pd
import numpy as np
import random
import sklearn.preprocessing as preprocessing
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

pd.set_option('display.height',1000)
pd.set_option('display.max_rows',30)
pd.set_option('display.max_columns',20)
pd.set_option('display.width',1000)

path = "/home/edda/tf/TensorflowExamples/DataSets/Kaggle/Titanic/"

train_data = pd.read_csv(path + "train.csv")
test_data = pd.read_csv(path + "test.csv")
x_train = ''
y_train = ''
x_test = ''
y_test = ''

def show_data():
    x_train = []
    y_train = []
    '''
    show data
    '''
    #print train_data.head()
    #print train_data.info()
    #print train_data.describe()
    #print train_data
    #print test_data

    #show_data_correlationship()
    #show_age_survived_distribution()
    #show_age_and_sex_distribution()
    #show_age_and_pclass_distribution()
    #show_pclass_and_survived_distribution()
    #show_pclass_and_age_and_survived_distribution()
    #show_age_and_sex_and_survived_distribution()
    #show_sex_and_pclass_and_survived_distribution()
    #show_fare_pclass_ditribution()
    #show_fare_and_survived_distribution()
    #show_sibsp_and_parch_distribution()
    #show_sibsp_and_parch_and_survived_distribution()
    #show_embarked_distribution()
    #show_embarked_and_age_distribution()
    #show_pclass_and_embarked_distribution()
    #plt.ion()
    #plt.tight_layout()
    #plt.show()
    #plt.pause(5)
    #plt.close()

def show_data_correlationship():
    sns.set(context="paper", font="monospace",style="white")
    f, axs = plt.subplots(figsize=(10,6))
    train_corr = train_data.drop("PassengerId", axis=1).corr()
    sns.heatmap(train_corr, ax=axs,vmax=0.9,square=True)
    axs.set_xticklabels(train_corr.index, size=15)
    axs.set_yticklabels(train_corr.columns[::-1], size=15)
    axs.set_title("train feature correlationship", fontsize=20)

def show_age_survived_distribution():
    f, axs = plt.subplots(2, 1, figsize=(8, 6))
    sns.set_style("white")
    sns.distplot(train_data.Age.fillna(-20), rug=True, color='b',ax=axs[0])

    axs0 = axs[0]
    axs0.set_title("Age distribution")
    axs0.set_xlabel('')

    axs1 = axs[1]
    axs1.set_title("Age survived distribution")
    axs1.set_xlabel('')

    sns.distplot(train_data[train_data.Survived == 0].Age.fillna(-20), color='r', hist=False, ax=axs1, label="dead")
    sns.distplot(train_data[train_data.Survived == 1].Age.fillna(-20), color='b', hist=False, ax=axs1, label="alive")

    axs1.legend(fontsize=16)

def show_age_and_sex_distribution():
    f, axs = plt.subplots(figsize=(8,3))
    axs.set_title("Age and sex")

    sns.distplot(train_data[train_data.Sex=='female'].dropna().Age, hist=False, color="r", label="Female")
    sns.distplot(train_data[train_data.Sex=='male'].dropna().Age, hist=False, color="b", label='male');
    axs.legend(fontsize=16)

def show_age_and_pclass_distribution():
    f, axs = plt.subplots(figsize=(8,3))
    axs.set_title("Age and Pclass")

    sns.distplot(train_data[train_data.Pclass==1].dropna().Age, hist=False, color='r', label='Pclass=1')
    sns.distplot(train_data[train_data.Pclass==2].dropna().Age, hist=False, color='g', label='Pclass=2')
    sns.distplot(train_data[train_data.Pclass==3].dropna().Age, hist=False, color='b', label='Pclass=3')

    axs.legend(fontsize=16)

def show_pclass_and_survived_distribution():
    y_dead = train_data[train_data.Survived == 0].groupby("Pclass")['Survived'].count()
    y_alive = train_data[train_data.Survived == 1].groupby("Pclass")['Survived'].count()

    pos = [1, 2, 3]
    axs = plt.figure(figsize=(8,4)).add_subplot(111)

    axs.bar(pos, y_dead, color="r", label="dead", alpha=0.5)
    axs.bar(pos, y_alive, color="b", bottom=y_dead, label="alive", alpha=0.5)
    axs.legend(loc="best", fontsize=16)
    axs.set_xticks(pos)
    axs.set_xticklabels(["Pclass%d"%(i) for i in range(1,4)], size=15)
    axs.set_title("Pclass survived count",size=20)

def show_pclass_and_age_and_survived_distribution():
    age_list = []
    for pclass in range(1, 4):
        for survived in range(0, 2):
            age_list.append(train_data[(train_data.Pclass == pclass) & (train_data.Survived == survived)].Age.dropna().values)

    f, axs = plt.subplots(3,1,figsize=(10, 10))

    pclas = 1
    for ax in axs:
        sns.distplot(age_list[pclas * 2 - 2], hist=False, ax=ax, label='Pclass={}, Survived=0'.format(pclas), color='r')
        sns.distplot(age_list[pclas * 2 - 1], hist=False, ax=ax, label='Pclass={}, Survived=1'.format(pclas), color='b')
        pclas += 1
        ax.set_xlabel("age", size=16)
        ax.legend(fontsize=10)

def show_age_and_sex_and_survived_distribution():
    f, axs = plt.subplots(2, 1, figsize=(8, 6))
    sns.distplot(train_data[(train_data.Sex == "male") & (train_data.Survived == 0)].dropna().Age, hist=False, rug=True, color='r', ax=axs[0], label="0")
    sns.distplot(train_data[(train_data.Sex == "male") & (train_data.Survived == 1)].dropna().Age, hist=False, rug=True, color='b', ax=axs[0], label="1")
    axs[0].set_title("male")
    axs[0].legend(fontsize=16)

    sns.distplot(train_data[(train_data.Sex == "female") & (train_data.Survived == 0)].Age.dropna(),hist=False, rug=True, color='r', ax=axs[1], label="0")
    sns.distplot(train_data[(train_data.Sex == "female") & (train_data.Survived == 1)].Age.dropna(),hist=False, rug=True, color='b', ax=axs[1], label="1")
    axs[1].set_title("female")
    axs[1].legend(fontsize=16)
    plt.tight_layout()

def show_sex_and_pclass_and_survived_distribution():
    label = []
    for sex in ["male", "female"]:
        for pclass in range(1, 4):
            label.append("sex:{}, pclass:{}".format(sex, pclass))

    pos = range(6)
    f = plt.figure(figsize=(16,8))
    ax = f.add_subplot(111)

    ax.bar(pos, train_data[train_data.Survived == 0].groupby(['Sex','Pclass']).Survived.count().values,
           color='r', alpha=0.5, align='center', tick_label=label, label='dead')

    ax.bar(pos, train_data[train_data.Survived == 1].groupby(['Sex', 'Pclass']).Survived.count().values,
           color='b', alpha=0.5, align='center', tick_label=label, label='dead',
           bottom=train_data[train_data.Survived == 0].groupby(['Sex','Pclass']).Survived.count().values)
    ax.tick_params(labelsize=15)
    ax.set_title("Sex Pclass Survived Distribution")
    ax.legend(fontsize=16, loc="best")

def show_fare_pclass_ditribution():
    f, axs = plt.subplots(2, 1, figsize=(8, 6))
    ax0 = axs[0]
    ax0.tick_params(labelsize=15)
    ax0.set_title("Fare distribution")
    ax0.set_ylabel("dist", size=15)
    sns.kdeplot(train_data.Fare, ax=ax0)
    sns.distplot(train_data.Fare, ax=axs[0])
    ax0.legend(fontsize=16)

    pos = range(0, 400, 50)
    ax0.set_xticks(pos)
    ax0.set_xlim([0, 200])
    ax0.set_xlabel('')

    ax1 = axs[1]
    ax1.set_title("Pclass Fare distribution")
    colors = ['r','g','b']
    for i in range(1, 4):
        sns.distplot(train_data[train_data.Pclass == i].Fare, color=colors[i - 1], label="Pclass={}".format(i), hist=False, ax=ax1)
    ax1.set_xlim([0, 200])
    ax1.legend(fontsize=16)

def show_fare_and_survived_distribution():
    f, ax = plt.subplots(figsize=(8, 3))
    ax.set_title("Fare Survived Distribution", size=20)
    sns.kdeplot(train_data[train_data.Survived == 0].Fare, color='r', label='dead', ax= ax)
    sns.kdeplot(train_data[train_data.Survived == 1].Fare, color='b', label='alive', ax=ax)
    ax.set_xlim(0, 300)
    ax.legend(fontsize=16)
    ax.set_xlabel("Fare", size=15)

def show_sibsp_and_parch_distribution():
    f, axs = plt.subplots(2, 1, figsize=(8,6))
    ax0 = axs[0]
    ax0.set_title("SibSp", size=20)
    sns.countplot(train_data.SibSp, color='b', ax=ax0)

    ax1 = axs[1]
    ax1.set_title("Parch", size=20)
    sns.countplot(train_data.Parch, color='b', ax=ax1)

def show_sibsp_and_parch_and_survived_distribution():
    f, axs = plt.subplots(3, 1, figsize=(8, 9))

    axs[0].set_title("SibSp Survived Rate", size=20)
    train_data.groupby("SibSp").Survived.mean().plot(kind='bar', ax=axs[0], color='r')
    axs[0].set_xlabel("")

    axs[1].set_title("Parch Survived Rate", size=20)
    train_data.groupby("Parch").Survived.mean().plot(kind='bar', ax=axs[1], color='g')
    axs[1].set_xlabel("")

    axs[2].set_title("Parch Add SibSp Survived Rate", size=20)
    train_data.groupby(train_data.SibSp + train_data.Parch).Survived.mean().plot(kind='bar', ax=axs[2], color='b')
    axs[2].set_xlabel("")

def show_embarked_distribution():
    plt.style.use("ggplot")
    f, axs = plt.subplots(figsize=(8, 3))
    y_dead = train_data[train_data.Survived == 0].groupby("Embarked")['Survived'].count().sort_index().values
    y_alive = train_data[train_data.Survived == 1].groupby("Embarked")['Survived'].count().sort_index().values
    pos = [1, 2, 3]

    axs.bar(pos, y_dead, color='r', alpha=0.5, align="center", label='dead')
    axs.bar(pos, y_alive, color='b', alpha=0.5, align="center", label='alive', bottom=y_dead)
    axs.set_xticks(pos)
    axs.set_title("Embard Survived Distribution", size=20)
    axs.legend(loc="best", fontsize=16)
    axs.set_xticklabels(['C','Q','S'])

def show_embarked_and_age_distribution():
    f, axs = plt.subplots(figsize=(8, 3))
    sns.distplot(train_data[train_data.Embarked == 'C'].Age.fillna(-20), ax=axs, color='r', hist=False, label='C')
    sns.distplot(train_data[train_data.Embarked == 'Q'].Age.fillna(-20), ax=axs, color='g', hist=False, label='Q')
    sns.distplot(train_data[train_data.Embarked == 'S'].Age.fillna(-20), ax=axs, color='b', hist=False, label='S')
    axs.set_title("Embarked Age Distribution")
    axs.legend(fontsize=16)
    axs.set_xlim(-20, 80)

def show_pclass_and_embarked_distribution():
    f, axs = plt.subplots(figsize=(8, 3))
    y_dead = train_data[train_data.Survived == 0].groupby(["Pclass","Embarked"]).Survived.count().reset_index().Survived.values
    y_alive = train_data[train_data.Survived == 1].groupby(['Pclass', 'Embarked']).Survived.count().reset_index().Survived.values

    xticklabels = []
    pos = range(9)
    for embarked in ['C', 'Q', 'S']:
        for pclass in range(1, 4):
            xticklabels.append("{}/{}".format(embarked, pclass))
    axs.bar(pos, y_dead, color='r', align='center', alpha=0.5, label='dead')
    axs.bar(pos, y_alive, color='b', align='center', alpha=0.5, label='alive', bottom=y_dead)
    axs.set_xticklabels(xticklabels)
    axs.set_xticks(pos)

def age_map(age):
    train_data.dropna()
    if age < 10:
        return '10-'
    if age < 60:
        return '{}-{}'.format(age//5 * 5, age//5 * 5 + 5)
    if age >= 60:
        return '60+'
    else:
        return 'NULL'

def basic_feature_engineering():
    #Embarked, NULL will be filled as 'S'
    train_data.Embarked.fillna('S', inplace=True)

    #Age will be segmented by 10
    train_data["Age_map"] = train_data['Age'].apply(lambda x:age_map(x))
    test_data["Age_map"] = test_data['Age'].apply(lambda x:age_map(x))

    #Fare, NULL will be filled as mean value
    test_data.loc[test_data.Fare.isnull(), 'Fare'] = \
        test_data[(test_data.Pclass == 1) & (test_data.Embarked == 'S') & (test_data.Sex == 'male')].dropna().Fare.mean()
    scaler = preprocessing.StandardScaler()
    fare_scaler_param = scaler.fit(train_data['Fare'].values.reshape(-1, 1))
    train_data.Fare = fare_scaler_param.transform(train_data.Fare.values.reshape(-1, 1))
    test_data.Fare = fare_scaler_param.transform(test_data.Fare.values.reshape(-1, 1))

    #Cabin, fill NULL as 0
    train_data.Cabin = train_data.Cabin.isnull().apply(lambda x: 'NULL' if x is True else 'Not NULL')
    test_data.Cabin = test_data.Cabin.isnull().apply(lambda x:'NULL' if x is True else 'Not NULL')

    #Do not consider name and ticket
    del train_data['Name'], train_data['Ticket']
    del test_data['Name'], test_data['Ticket']

    #Transform all data to onehot
    x_train = pd.concat([train_data[['SibSp', 'Parch', 'Fare']], pd.get_dummies(train_data[['Pclass','Sex','Cabin','Embarked','Age_map']])], axis=1)
    y_train = train_data['Survived']

    x_test = pd.concat([test_data[['SibSp','Parch','Fare']], pd.get_dummies(test_data[['Pclass', 'Sex', 'Cabin', 'Embarked', 'Age_map']])], axis=1)
    return x_train, y_train, x_test

def basic_model():
    model = LogisticRegression()
    param = {'penalty':['l1', 'l2'],
             'C':[0.1, 0.5, 1.0, 5.0]}
    grd = GridSearchCV(estimator=model, param_grid=param, cv=5, n_jobs=3)
    grd.fit(x_train, y_train)


    train_sizes = np.linspace(0.05, 1.0, 5)
    train_size, train_scores, test_scores = learning_curve(grd, x_train, y_train, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax = plt.figure().add_subplot(111)
    ax.set_title("Learning Rate")
    ax.set_xlabel("Training Samples Number")
    ax.set_ylabel("Score")

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='b')
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='r')

    ax.plot(train_sizes, train_scores_mean, 'x-', color='b', label='Train Score')
    ax.plot(train_sizes, test_scores_mean, 'x-', color='r', label='Test Score')



    ax.legend(loc='best')
    #plt.ion()
    #plt.tight_layout()
    #plt.show()
    #plt.pause(5)
    #plt.close()

    gender_submission = pd.DataFrame({"PassengerId":test_data.iloc[:,0], "Survived":grd.predict(x_test)})
    gender_submission.to_csv(path + "gender_submission.csv", index=None)

if __name__ == "__main__":
    #show_data()
    x_train, y_train, x_test = basic_feature_engineering()
    basic_model()
