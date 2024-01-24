import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

import optuna

df = pd.read_csv('voice.csv')
df.head()
df.isnull().sum()

def unique(col):
    return df[col].nunique()

d = {}
for i in df.columns:
    value = unique(i)
    d[i] = value
print(d, sep="\n")

x = df.iloc[:, :-1]
y = df.label

x.head()
y.head()

df['label'].value_counts()
data = ['Female', 'Male']
d = [1584, 1584]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# split dataset into training and test with test size as 20 percent

# 1) Decision Tree Classifier
def treeobjective(trial):
    random_state = trial.suggest_int("random_state", 1, 64, log=True)
    criterion = trial.suggest_categorical("criterion", ['gini', 'entropy',  'log_loss'])
    splitter = trial.suggest_categorical("splitter", ['best', 'random'])

    DTmodel = DecisionTreeClassifier(random_state=random_state, criterion=criterion, splitter=splitter)
    DTmodel.fit(x_train, y_train)
    score = DTmodel.score(x_test, y_test)
    return score

tree_study = optuna.create_study(direction="maximize")
tree_study.optimize(treeobjective, n_trials=250)

tree_trial = tree_study.best_trial
best_params_tree = tree_trial.params
print('Accuracy: {}'.format(tree_trial.value))
print("Best hyperparameters: {}".format(tree_trial.params))

fig = optuna.visualization.plot_optimization_history(tree_study)
fig.show()
fig2 = optuna.visualization.plot_slice(tree_study)
fig2.show()

DTmodel = DecisionTreeClassifier(random_state=best_params_tree['random_state'], criterion=best_params_tree['criterion'], splitter=best_params_tree['splitter'])
DTmodel.fit(x_train, y_train)
y_pred1 = DTmodel.predict(x_test)
y_pred1 = np.array(y_pred1)
print(y_pred1[:10])
print('score of decision tree model is: ', DTmodel.score(x_test, y_test))
pd.crosstab(y_pred1, y_test, rownames=['matrix'], colnames=['confusion'], margins=True)
print("\t\t\tDecision Tree Class report:\n", classification_report(y_pred1, y_test))
print("Decision Tree Accuracy score: ", accuracy_score(y_pred1, y_test) * 100, "%")

# 2) ANN
scaler = StandardScaler()
le = LabelEncoder()
x_train2 = scaler.fit_transform(x_train)
x_test2 = scaler.fit_transform(x_test)
y_train2 = le.fit_transform(y_train)
y_test2 = le.fit_transform(y_test)

def objective(trial):
    neurons = trial.suggest_int("neurons", 8, 64, log=True)
    layers = trial.suggest_int("layers", 1, 8, log=True)
    activation = trial.suggest_categorical("activation_function", ['sigmoid', 'relu',  'softmax'])
    epochs = trial.suggest_int("epochs", 10, 150, log=True)
    dropout = trial.suggest_uniform("dropout_rate", 0, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)

    ANNmodel = Sequential()
    ANNmodel.add(Dense(neurons, input_dim=20, activation=activation))
    while layers > 0:
        ANNmodel.add(Dense(neurons, activation=activation))
        ANNmodel.add(Dropout(dropout))
        layers -= 1
    ANNmodel.add(Dense(1, activation='sigmoid'))

    # binary_crossentropy is used instead of categorical_crossentropy because there are only two catagories male/female if we had more we would have had to use categorical
    ANNmodel.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate), metrics=['accuracy'])
    ANNmodel.fit(x_train2, y_train2, epochs=epochs, batch_size=32, verbose=0)
    score = ANNmodel.evaluate(x_test2, y_test2)
    return score[1]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=250)

trial = study.best_trial
best_params = trial.params

print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))

fig = optuna.visualization.plot_optimization_history(study)
fig.show()
fig2 = optuna.visualization.plot_slice(study)
fig2.show()

ANNmodel = Sequential()
ANNmodel.add(Dense(best_params["neurons"], input_dim=20, activation=best_params["activation_function"]))
layers = best_params["layers"]
while layers > 0:
    ANNmodel.add(Dense(best_params["neurons"], activation=best_params["activation_function"]))
    ANNmodel.add(Dropout(best_params["dropout_rate"]))
    layers -= 1
ANNmodel.add(Dense(1, activation='sigmoid'))

# binary_crossentropy is used instead of categorical_crossentropy because there are only two catagories male/female if we had more we would have had to use categorical
ANNmodel.compile(loss='binary_crossentropy', optimizer=Adam(best_params["learning_rate"]), metrics=['accuracy'])

ANNmodel.fit(x_train2, y_train2, epochs=best_params["epochs"], batch_size=32, verbose=0)
# Making predictions
y_pred2 = ANNmodel.predict(x_test2)
y_pred2num = (y_pred2 > 0.5).astype(int)

# Printing Results
print('score of Neural Network model is: ', ANNmodel.evaluate(x_test2, y_test2)[1])
print("\t\t\tNeural Network Class report:\n", classification_report(y_pred2num, y_test2))
print("Neural Network Accuracy score: ", accuracy_score(y_pred2num, y_test2) * 100, "%")

y_pred2 = np.where(y_pred2 > 0.5, 'male', 'female')
y_pred2 = np.squeeze(y_pred2)
print(y_pred2[:10])
pd.crosstab(y_pred2, y_test2, rownames=['matrix'], colnames=['confusion'], margins=True)

# 3)Logistic Regression
def logobjective(trial):
    regularization_penalty = trial.suggest_categorical("regularization_penalty", ['l2',  None])
    max_iterations = trial.suggest_int("meximum_iterations", 100, 5000, log=True)
    solver = trial.suggest_categorical("solver", ['newton-cg', 'sag', 'saga'])

    LRmodel = LogisticRegression(max_iter=max_iterations, penalty=regularization_penalty, solver=solver)
    LRmodel.fit(x_train, y_train)
    score = LRmodel.score(x_test, y_test)
    return score
log_study = optuna.create_study(direction="maximize")
log_study.optimize(logobjective, n_trials=250)

log_trial = log_study.best_trial
best_params_log = log_trial.params
print('Accuracy: {}'.format(log_trial.value))
print("Best hyperparameters: {}".format(log_trial.params))

fig = optuna.visualization.plot_optimization_history(log_study)
fig.show()
fig2 = optuna.visualization.plot_slice(log_study)
fig2.show()

LRmodel = LogisticRegression(max_iter=best_params_log["meximum_iterations"], penalty=best_params_log["regularization_penalty"], solver=best_params_log["solver"])
LRmodel.fit(x_train, y_train)
y_pred3 = LRmodel.predict(x_test)
print(y_pred3[:10])
print('score of Log Reg model is: ', LRmodel.score(x_test, y_test))
pd.crosstab(y_pred3, y_test, rownames=['matrix'], colnames=['confusion'], margins=True)
print("\t\t\tLog Reg Class report:\n", classification_report(y_pred3, y_test))
print("Log Reg Accuracy score: ", accuracy_score(y_pred3, y_test) * 100, "%")

# 4) KNN Classifier
def knnobjective(trial):

    n_neighbors = trial.suggest_int("n_neighbors", 1, 25, log=True)
    regularization = trial.suggest_int("p", 1, 2, log=True)
    weight = trial.suggest_categorical("weight", ['uniform', 'distance'])

    KNmodel = KNeighborsClassifier(n_neighbors=n_neighbors, metric='minkowski', p=regularization, weights=weight )
    KNmodel.fit(x_train, y_train)
    score = KNmodel.score(x_test, y_test)
    return score
knn_study = optuna.create_study(direction="maximize")
knn_study.optimize(knnobjective, n_trials=250)

knn_trial = knn_study.best_trial
best_params_knn = knn_trial.params
print('Accuracy: {}'.format(knn_trial.value))
print("Best hyperparameters: {}".format(knn_trial.params))

fig = optuna.visualization.plot_optimization_history(knn_study)
fig.show()
fig2 = optuna.visualization.plot_slice(knn_study)
fig2.show()

KNmodel = KNeighborsClassifier(n_neighbors=best_params_knn["n_neighbors"], p=best_params_knn["p"], weights=best_params_knn["weight"])
KNmodel.fit(x_train, y_train)
y_pred4 = KNmodel.predict(x_test)
print(y_pred4[:10])
print('score of KNN model is: ', KNmodel.score(x_test, y_test))
pd.crosstab(y_pred4, y_test, rownames=['matrix'], colnames=['confusion'], margins=True)
print("\t\t\tKNN report:\n", classification_report(y_pred4, y_test))
print("KNN Accuracy score: ", accuracy_score(y_pred4, y_test) * 100, "%")

# 5) SVM Model
def svmobjective(trial):

    gamma = trial.suggest_categorical("gamma", ['scale', 'auto'])
    # poly took way too long to compute
    kernal = trial.suggest_categorical("kernal", ['linear', 'rbf', 'sigmoid'])

    SVMmodel = SVC(kernel=kernal, gamma=gamma)
    SVMmodel.fit(x_train, y_train)
    score = SVMmodel.score(x_test, y_test)
    return score
svm_study = optuna.create_study(direction="maximize")
svm_study.optimize(svmobjective, n_trials=20)

svm_trial = svm_study.best_trial
best_params_svm = svm_trial.params
print('Accuracy: {}'.format(svm_trial.value))
print("Best hyperparameters: {}".format(svm_trial.params))

fig = optuna.visualization.plot_optimization_history(svm_study)
fig.show()
fig2 = optuna.visualization.plot_slice(svm_study)
fig2.show()

SVMmodel = SVC(kernel=best_params_svm['kernal'], gamma=best_params_svm['gamma'])
SVMmodel.fit(x_train, y_train)
y_pred5 = SVMmodel.predict(x_test)
print(y_pred5[:10])
print('score of SVM model is: ', SVMmodel.score(x_test, y_test))
print(pd.crosstab(y_pred5, y_test, rownames=['matrix'], colnames=['confusion'], margins=True))
print("\t\t\tSVM report:\n", classification_report(y_pred5, y_test))
print("SVM Accuracy score: ", accuracy_score(y_pred5, y_test) * 100, "%")

# Print all Predictions
list1 = [y_pred1, y_pred2, y_pred3, y_pred4, y_pred5]
params_list = best_params_tree, best_params, best_params_log, best_params_knn, best_params_svm
d = ['DecTree', 'ANN', 'Log Regression', 'KNN', 'SuppVecMachine']
a = {}
k = 0
list2 = []

for i in list1:
    list2.append(accuracy_score(i, y_test) * 100)
for i in d:
    a[i] = list2[k]
    k += 1

print("List of all model accuracies:\n", a)
print("List of best parameters:\n", params_list)
print("the most accurate model is:", max(a, key=a.get),)
