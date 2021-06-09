# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 02:47:29 2019

@author: Hanaa
"""


#importing libraries
import numpy as np
#import matplotlib.pyplot as pyplot
import pandas as pd
# for correlatio
import seaborn as sns

from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
#from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score, roc_curve, auc 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble
import xgboost as xgb
import lightgbm as lgb
from skopt import BayesSearchCV
#from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

###########------
# prepocessing pipline
####-----------
##################
"""
class Scores():
    def __init__(self, dataset=None, labels=None, clf=None):
        
        if dataset is None or labels is None or clf is None:
            self.roc = 0
            self.accuracy = 0
            self.precision = 0
            self.recall = 0
            self.f1 = 0
            self.cm = None
        
        else:

            _, _, self.roc = calculate_roc_score(clf, dataset, labels)

            predictions = np.array(clf.predict(dataset), dtype=np.int32)
            labels = np.array(labels, dtype=np.int32)
            scores = precision_recall_fscore_support(labels, predictions, average='binary')
            self.accuracy = accuracy_score(labels, predictions)
            self.precision = scores[0]
            self.recall = scores[1]
            self.f1 = scores[2]
            self.cm = confusion_matrix(labels, predictions)
        
    def __str__(self):
        return 'ROC AUC: {:.3f}\nPrecision: {:.3f}\nRecall: {:.3f}\nF1: {:.3f}'.format(self.roc, self.precision,
                                                                                                       self.recall, self.f1)
    def __iadd__(self, other):
        self.roc += other.roc
        self.accuracy += other.accuracy
        self.precision += other.precision
        self.recall += other.recall
        self.f1 += other.f1
        return self
        
    def __itruediv__(self, other):
        self.roc /= other
        self.accuracy /= other
        self.precision /= other
        self.recall /= other
        self.f1 /= other
        return self
####################    

#Define a TIme Class to computer total execution time
import time
class Timer:
  def __init__(self):
    self.start = time.time()

  def restart(self):
    self.start = time.time()

  def get_time(self):
    end = time.time()
    m, s = divmod(end - self.start, 60)
    h, m = divmod(m, 60)
    time_str = "%02d:%02d:%02d" % (h, m, s)
    return time_str

####Define a method to print the Confusion Matrix and the performance metrics
def Print_confusion_matrix(cm, auc, heading):
    print('\n', heading)
    print(cm)
    true_negative  = cm[0,0]
    true_positive  = cm[1,1]
    false_negative = cm[1,0]
    false_positive = cm[0,1]
    total = true_negative + true_positive + false_negative + false_positive
    accuracy = (true_positive + true_negative)/total
    precision = (true_positive)/(true_positive + false_positive)
    recall = (true_positive)/(true_positive + false_negative)
    misclassification_rate = (false_positive + false_negative)/total
    F1 = (2*true_positive)/(2*true_positive + false_positive + false_negative)
    print('accuracy.................%7.4f' % accuracy)
    print('precision................%7.4f' % precision)
    print('recall...................%7.4f' % recall)
    print('F1.......................%7.4f' % F1)
    print('auc......................%7.4f' % auc)

##### Plot the learning curves
from sklearn.model_selection import learning_curve, ShuffleSplit
def Plot_learning_curve(estimator, title, X, y, ylim = None, cv = None,
                        n_jobs = 1, train_sizes = np.linspace(0.1, 1.0, 5)):
    pyplot.figure()
    pyplot.title(title)
    if ylim is not None:
        pyplot.ylim(*ylim)
    pyplot.xlabel("Training examples")
    pyplot.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, 
                                                            X, y,
                                                            cv = cv,
                                                            n_jobs = n_jobs,
                                                            train_sizes = train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    pyplot.grid()

    pyplot.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    pyplot.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    pyplot.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    pyplot.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    pyplot.legend(loc="best")
    pyplot.show()
    return

# Sensitivity
def sensitivity(model,y_true, y_pred):
    y_pred=math_ops.round(y_pred)
    TP = model.count_nonzero(y_pred * y_true)
    TN = model.count_nonzero((y_pred - 1) * (y_true - 1))
    FP = model.count_nonzero(y_pred * (y_true - 1))
    FN = model.count_nonzero((y_pred - 1) * y_true)
    metric=tf.divide(TP,TP+FN)
    return metric

# Specificity
def specificity(model, y_true,y_pred):
    y_pred=math_ops.round(y_pred)
    TP = model.count_nonzero(y_pred * y_true)
    TN = model.count_nonzero((y_pred - 1) * (y_true - 1))
    FP = model.count_nonzero(y_pred * (y_true - 1))
    FN = model.count_nonzero((y_pred - 1) * y_true)
    metric=tf.divide(TN,TN+FP)
    return metric


################# Roc curve
def plot_roc_curve(fpr, tpr):
    pyplot.plot(fpr, tpr, color='orange', label='ROC')
    pyplot.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title('Receiver Operating Characteristic (ROC) Curve')
    pyplot.legend()
    pyplot.show() 
################### Later for medical data
#https://machinelearningmastery.com/how-to-prepare-categorical-data-for-deep-learning-in-python/
#####################  
    
"""    
#############importing dataset
datasetT = pd.read_csv('C:/__MyData/___________PredictingMultipleSclerosisDisease/Dataset-t.csv')

datasets = np.transpose(datasetT)

X = datasets.iloc[: , :-1].values
Y = datasets.iloc[: , 18720].values
y= Y.astype('int')
############# nomalizarion
from sklearn.preprocessing import Normalizer
X = Normalizer().fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(np.nan_to_num(X),y,test_size = 0.2,random_state=0)



#solving problem of missing data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#impurter = imputer.fit(X[: , :])
#X[: , :] = imputer.transform(X[: , :])

#col_mask=df.isnull().any(axis=0) 
#pd.DataFrame(X_train).fillna(X_train.mean())
#pd.DataFrame(X_test).fillna(X_test.mean())

#################   feature scaling
#from sklearn.preprocessing import StandardScaler
#sc_x = StandardScaler()
#X_train = sc_x.fit_transform(X_train)
#X_test = sc_x.transform(X_test)

########################### features correlations----- to be done afet selection
#f, ax = pyplot.subplots(figsize=(13, 13))
#corr = pd.DataFrame(X).corr(method='pearson')
#sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
#            cmap=sns.diverging_palette(220, 10, as_cmap=True),
#            square=True, ax=ax)
###########
#############  1-  PCA reduction, the number of componatat are chnaged manualy by Hanaa
from sklearn.decomposition import PCA as sklearnPCA
from sklearn import preprocessing

sklearn_pca = sklearnPCA(n_components=100)

fit = sklearn_pca.fit(X)
X_t_train = sklearn_pca.transform(X_train)
X_t_test = sklearn_pca.transform(X_test)

###########
scaled = pd.DataFrame(preprocessing.scale(X_t_train))
scaled.plot(kind="hist", legend=None, bins=20, color='k')
scaled.plot(kind="kde", legend=None);
"""
############## corrolation for new features
sns.pairplot(pd.DataFrame(X_t_train[0:,0:], index=[i for i in range(X_t_train.shape[0])], columns=['f'+str(i) for i in range(X_t_train.shape[1])]))
pyplot.title('Pairplot for the Data', fontsize = 20)
pyplot.savefig('correlation1');

###############
cum_sum = sklearn_pca.explained_variance_ratio_.cumsum()
cum_sum = cum_sum*100

fix, ax = pyplot.subplots(figsize = (8, 8))
pyplot.bar(range(20), cum_sum, color = 'r',alpha=0.5)
pyplot.title('PCA Analysis')
pyplot.ylabel('cumulative explained variance')
pyplot.xlabel('number of components')
pyplot.locator_params(axis='y', nbins=20)

# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)

########### pipline with LR
from sklearn.pipeline import Pipeline
def pipeline_PCA_GLM(components):
    accuracy_chart = []
    for i in components:
        steps = [('pca', sklearnPCA(n_components = i)),
        ('estimator', LogisticRegression())]
        pipe = Pipeline(steps)
        pipe.fit(X_train, Y_train)
        predictions = pipe.predict(X_test)
        accuracy_chart.append(accuracy_score(Y_test,predictions))
    return accuracy_chart
n_components = range(0,200, 10)
accuracy_chart = pipeline_PCA_GLM(n_components)

pyplot.figure(figsize=(10, 8))
pyplot.bar(n_components, accuracy_chart)
pyplot.ylim(0,1)
pyplot.xlim(0,200)
pyplot.locator_params(axis='y', nbins=20)
pyplot.locator_params(axis = 'x', nbins = 20)
pyplot.ylabel("Accuracy")
pyplot.xlabel("Number of Components")
########################### features correlations----- 
#Correlation Graph
f, ax = pyplot.subplots(figsize=(13, 13))
corr = pd.DataFrame(X_t_train).corr(method='pearson')
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


#######################
"""
############### Univariate selection

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# feature extraction
test = SelectKBest(score_func=chi2, k=100)
fit = test.fit(X, Y)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
X_u_train = fit.transform(X_train)
X_u_test = fit.transform(X_test)
# summarize selected features
#print(X_u_train[0:5,:])
"""
###############3
scaled = pd.DataFrame(preprocessing.scale(X_u_train))
scaled.plot(kind="hist", legend=None, bins=20, color='k')
scaled.plot(kind="kde", legend=None);
##########Correlation Graph
f, ax = pyplot.subplots(figsize=(13, 13))
corr = pd.DataFrame(X_u_train).corr(method='pearson')
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

# ##################Could be tested on the differentially expressed gene############################
# Univariate feature selection with F-test for feature scoring
# We use the default selection function: the 10% most significant features

from sklearn.feature_selection import SelectPercentile, f_classif 
X_indices = np.arange(X.shape[-1])
selector = SelectPercentile(f_classif, percentile=10)
select = selector.fit(X, Y)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
pyplot.bar(X_indices - .45, scores, width=.2,
        label=r'Univariate score ($-Log(p_{value})$)', color='darkorange',
        edgecolor='black')

# #######################################################
# the weights of an SVM
clf = SVC(kernel='linear')
clf.fit(X_u_train, Y_train)
X_indices = np.arange(X_u_train.shape[-1])
svm_weights = (clf.coef_ ** 2).sum(axis=0)
svm_weights /= svm_weights.max()

pyplot.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight',
        color='navy', edgecolor='black')
################# Compare to the weights of an SVM with and without T-test selection
clf_selected = SVC(kernel='linear')
clf_selected.fit(selector.transform(X), Y)
X_indices = np.arange(X.shape[-1])
svm_weights_selected = (clf_selected.coef_ ** 2).sum(axis=0)
svm_weights_selected /= svm_weights_selected.max()

pyplot.bar(X_indices[selector.get_support()] - .05, svm_weights_selected,
        width=.2, label='SVM weights after selection', color='c',
        edgecolor='black')

pyplot.title("Comparing feature selection")
pyplot.xlabel('Feature number')
pyplot.yticks(())
pyplot.axis('tight')
pyplot.legend(loc='upper right')
pyplot.show()
########################### features correlations
#################

"""
#################   
####################### 3-	Recursive feature elimination
from sklearn.feature_selection import RFE
# feature extraction
#modelr = LogisticRegression(solver='lbfgs')
modelr = SVC(random_state=10, probability=True)
modelr = LinearSVC()

rfe = RFE(modelr, 3)
fit = rfe.fit(X, Y)

print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

X_r_train = fit.transform(X_train)
X_r_test = fit.transform(X_test)
"""
########################### features correlations
#Correlation Graph
f, ax = pyplot.subplots(figsize=(13, 13))
corr = pd.DataFrame(X_r_train).corr(method='pearson')
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

####################################
# Cross validation

# Logit wieghts

def plot_logit_weights(clf_logit, title):
    pyplot.figure()
    pyplot.title(title)
    pyplot.plot(np.arange(clf_logit.coef_.shape[1]), clf_logit.coef_[0])
    pyplot.show()


def plot_logit_weights_ax(ax, clf_logit, title):
    ax.set_title(title)
    sns.lineplot(np.arange(clf_logit.coef_.shape[1]), clf_logit.coef_[0], ax=ax)
    
#####################    
clf_svm = LinearSVC(penalty='l1', dual=False, max_iter=10000, random_state=10, C=0.034081632653061224)
clf_svm = fit_clf_print_scores(clf_svm, X_train, Y_train, X_test, Y_test)

clf_rfe = RFE(clf_svm, n_features_to_select=1, step=1, verbose=1)
clf_rfe.fit(X_train, Y_train)
np.save('./../selection_results/svc_rfe_ranking.npy', clf_rfe.ranking_, allow_pickle=False)
####################################################################

"""
#####################
######################### 2-	Feature Selection with XGBoost (Feature Importance)
# Use feature importance for feature selection

from numpy import sort
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

# feature selction model on all data
model = XGBClassifier()

selector = SelectFromModel(estimator=model).fit(X, Y)
# Make predictions for test data and evaluate
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
predictions = [value for value in y_pred]
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

############## selection
X_x_train = selector.transform(X_train)
X_x_test = selector.transform(X_test)

###Fit model on the selected features---- to be modefied for gene names
model.fit(X_x_train, Y_train)
# plot feature importance
plot_importance(model)
pyplot.show()

"""
# Make predictions for test data and evaluate
y_pred = model.predict(X_x_test)
predictions = [value for value in y_pred]
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# ############# another boosting algorithm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
# Initialize and fit GBRT
gbrt = GradientBoostingRegressor(max_depth=5, n_estimators=120)
gbrt.fit(X, Y)

# Measure the validation error at each stage of training
errors = [mean_squared_error(Y, y_pred) for y_pred in gbrt.staged_predict(X)]

# Find the optimal number of trees
best_n_estimators = np.argmin(errors)

# Train another GBRT ensemble using the optimal number of trees
gbrt_best = GradientBoostingRegressor(max_depth=5, n_estimators=best_n_estimators)

gbrt_best.fit(X_train, Y_train)
y_pred = gbrt_best.predict(X_test)
predictions = [value for value in y_pred]
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

############## fit on all feature to selection
gbrt_best.fit(X, Y)
X_x2_train = gbrt_best.transform(X_train)
X_x2_test = gbrt_best.transform(X_test)

###Fit model on the selected features---- to be modefied for gene names
gbrt_best.fit(X_x2_train, Y_train)
# plot feature importance
plot_importance(gbrt_best)
pyplot.show()

# Make predictions for test data and evaluate
y_pred = gbrt_best.predict(X_x2_test)
predictions = [value for value in y_pred]
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#######   Select feature based on Importance  ######https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
#thresholds = sort(model.feature_importances_)
#print(thresholds)
#thresholds.tofile('C:/__MyData/___________PredictingMultipleSclerosisDisease/feature_importances.csv',sep=',',format='%10.5f')

#import csv
#with open('feature_importances.csv','wb') as result_file:
#    wr = csv.writer(result_file, dialect='excel')
#    wr.writerows(thresholds)   
#for thresh in thresholds:
#	# select features using threshold
#	selection = SelectFromModel(model, threshold=thresh, prefit=True)
#	select_X_train = selection.transform(X_train)
	# train model
#	selection_model = XGBClassifier()
#	selection_model.fit(select_X_train, Y_train)
	# eval model
#	select_X_test = selection.transform(X_test)
#	y_pred = selection_model.predict(select_X_test)
#	predictions = [value for value in y_pred]
#	accuracy = accuracy_score(Y_test, predictions)
#	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))

########################### features correlations
#Correlation Graph
f, ax = pyplot.subplots(figsize=(13, 13))
corr = pd.DataFrame(X_x_train).corr(method='pearson')
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

################
###  soft voting ensemple with Pre-fitted Classifiers
################33
from sklearn.ensemble import VotingClassifier

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()

for clf in (clf1, clf2, clf3):
    clf.fit(X, Y)
    
estimators = [('lr', clf1), ('rf', clf2), ('svc', clf3)]

modelv = VotingClassifier(estimators=estimators, weights=[1,1,1], voting='soft')

#labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'Ensemble']
modelv.fit(X, Y)

X_v_train = modelv.transform(X_train)
X_v_test = modelv.transform(X_test)

###Fit model on the selected features
modelv.fit(X_v_train, Y_train)
# Make predictions for test data and evaluate
y_pred = modelv.predict(X_v_test)
predictions = [value for value in y_pred]
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#####
########## Correlation Matrix------ to be modified after DEG and selection
cmap=sns.diverging_palette(5, 250, as_cmap=True)

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "10pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "11pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '11pt')])]

corr.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '8pt'})\
    .set_caption('Correlation Matrix')\
    .set_precision(2)\
    .set_table_styles(magnify())
    
############## Plot feature importance
pyplot.subplots(figsize=(13, 6))
pyplot.title('Feature ranking', fontsize = 18)
pyplot.ylabel('Importance degree', fontsize = 13)

feature_names = pd.DataFrame(X).columns
pyplot.xticks(range(X.shape[1]), feature_names, fontsize = 9)
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()

########

"""

######################33 
############ create all the machine learning models
################################
scoring    = "accuracy"
num_trees = 100
test_size = 0.20
seed      = 10
models = []
models.append(('LR', LogisticRegression(random_state=seed)))
models.append(('Proposed Method', LinearDiscriminantAnalysis()))
models.append(('Proposed Method', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))
models.append(('Proposed Method', MLPClassifier()))
models.append(('RF', ExtraTreesClassifier())) ### Ensempl calssifier
models.append(('SVM', SVC(random_state=seed, probability=True)))
models.append(('KNN', DecisionTreeClassifier(random_state=seed)))
models.append(('KNN', KNeighborsClassifier()))

################# ROC Curve
import pandas as pd
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score, roc_curve, auc 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_curve, roc_auc_score

ROCresult_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

# Train the models and record the results
for name, model in models:
    classifier = model.fit(X_x_train, Y_train)
    yproba = classifier.predict_proba(X_x_test)[::,1]   
    fpr, tpr, _ = roc_curve(Y_test,  yproba)
    auc = roc_auc_score(Y_test, yproba)   
    ROCresult_table = ROCresult_table.append({'classifiers':name,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)

# Set name of the classifiers as index labels
ROCresult_table.set_index('classifiers', inplace=True)
##########3 draw
fig = plt.figure()
#ax = plt.axes()
#ax.set_facecolor("white")
for i in ROCresult_table.index:
    plt.plot(ROCresult_table.loc[i]['fpr'], 
             ROCresult_table.loc[i]['tpr'], 
             label="{}, AUC={:.2f}".format(i, ROCresult_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='grey', linestyle='--')
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)
plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)
plt.title('ROC curve with DEGs ', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')
plt.show()

######



# 10-fold cross validation all model at once
# variables to hold the results and names
#"""
from sklearn.metrics import precision_score, recall_score

results = []
names   = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_x_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
####### boxplot algorithm comparison accuracy
fig = pyplot.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

#"""


######################
"""
###########3 not working yet.
    # Compute and return the:
    # F1 scores (the harmonic mean of precision and recall)
    # Precision scores
    # Recall scores as separate dictionaries   
f1_dct={}
precision_dct={}
recall_dct={}
for name, model in models:
    model.fit(X_t_train, Y_train)
    predicted = model.predict(X_t_test)
    f1_dct[model]=round(f1_score(Y_test, predicted, pos_label='edible'), 4)
    precision_dct[model]= round(precision_score(Y_test, predicted, pos_label='edible'), 4)
    recall_dct[model]= round(recall_score(Y_test, predicted, pos_label='edible'), 4)

print("Precision", sorted(precision_dct.items(), key=lambda x:x[1], reverse=True))           
print("Recall", sorted(recall_dct.items(), key=lambda x:x[1], reverse=True))            
print("F1-score", sorted(f1_dct.items(),key=lambda x:x[1], reverse=True))


########### Classifiactio RF Classification
#classifier1 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
#fitting the classifier to the training set
classifier2 = RandomForestClassifier(n_estimators = 100, criterion='entropy', random_state=0)
######################
classifier2.fit(X_t_train, Y_train)

############### cross validation
#kfold = KFold(n_splits=10, shuffle=True random_state=10)
#cv_results = cross_val_score(classifier2, X_u_train, Y_train, cv=kfold, scoring=scoring)
#results = cross_val_score(classifier2,sc_x.fit_transform(X), Y, cv=kfold, scoring='neg_log_loss', n_jobs=-1)

# Predicting the Test set results
Y_pred = classifier2.predict(X_t_test)

##########33   Roc curve
print ("Roc curve:")
probs = classifier2.predict_proba(X_t_test)
#Keep Probabilities of the positive class only.
probs = probs[:, 1]
#Compute the AUC Score.
auc = roc_auc_score(Y_test, probs)
print('AUC: %.2f' % auc)
#Get the ROC Curve.
fpr, tpr, thresholds = roc_curve(Y_test, probs,1)
plot_roc_curve(fpr, tpr)


#########classification report
c_report = classification_report(Y_test, Y_pred)
print('\nClassification report:\n', c_report)

######### kappa score Cohen’s kappa: Cohen’s kappa statistics measures how agreeable the prediction 
#and the true label are. It ranges between -1 (completely disagree) 
#and 1 (completely agree)
"""
kappa_report = cohen_kappa_score(Y_test, Y_pred)
print('\nkappa score:\n', kappa_report)

########### Confiusion matrix --------- not working
y_predicted_test = classifier.predict(X_t_test)
cm = confusion_matrix(Y_test, y_predicted_test)
AUC = roc_auc_score(Y_test, y_predicted_test)
Print_confusion_matrix(cm, AUC, 'Confusion matrics of the test dataset')

############ senstivity and Specificity
sen = sensitivity(classifier2,Y_test, Y_pred)
spec = specificity(model, Y_test, Y_pred)
print("sensitivity: %.2f%%" % (sen * 100.0))
print("specificity: %.2f%%" % (spec * 100.0))
###############
####### calculate bier loss score, It takes the true class values (0, 1) and the predicted 
#probabilities for all examples in a test dataset as arguments and returns the average Brier score
# predict probabilities
probs = model.predict_proba(X_t_test)
# keep the predictions for class 1 only
probs = probs[:, 1]
losses = brier_score_loss(Y_test, probs)
# plot input to loss
pyplot.plot(probs, losses)
pyplot.show()
###
########### Learning curve
y_train = np.reshape(Y_train, len(Y_train))
cv_ = ShuffleSplit(n_splits = 20, test_size = 0.20, random_state = 0)
Plot_learning_curve(classifier, 'Learning Curves', X_t_train, y_train, cv = cv_, n_jobs = 1)


# #####################################################
# Plot calibration plots
import numpy as np
np.random.seed(0)

import matplotlib.pyplot as pyplot

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split

X, y = datasets.make_classification(n_samples=100000, n_features=20,
                                    n_informative=2, n_redundant=2)

train_samples = 100  # Samples used for training the models

X_train = X[:train_samples]
X_test = X[train_samples:]
y_train = y[:train_samples]
y_test = y[train_samples:]

# Create classifiers
lr = LogisticRegression(solver='lbfgs')
gnb = GaussianNB()
svc = LinearSVC(C=1.0)
rfc = RandomForestClassifier(n_estimators=100)
pyplot.figure(figsize=(10, 10))
ax1 = pyplot.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = pyplot.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for clf, name in [(lr, 'Logistic'),
                  (gnb, 'Naive Bayes'),
                  (svc, 'Support Vector Classification'),
                  (rfc, 'Random Forest')]:
    clf.fit(X_train, y_train)
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(X_test)[:, 1]
    else:  # use decision function
        prob_pos = clf.decision_function(X_test)
        prob_pos = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y_test, prob_pos, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (name, ))

    ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
             histtype="step", lw=2)

ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots  (reliability curve)')

ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="upper center", ncol=2)

pyplot.tight_layout()
pyplot.show()

def plot_calibration_curve(est, name, fig_index):
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1., solver='lbfgs')

    fig = pyplot.figure(fig_index, figsize=(10, 10))
    ax1 = pyplot.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = pyplot.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                      (est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    pyplot.tight_layout()

# Plot calibration curve for Gaussian Naive Bayes
plot_calibration_curve(GaussianNB(), "Naive Bayes", 1)

# Plot calibration curve for Linear SVC
plot_calibration_curve(LinearSVC(max_iter=10000), "SVC", 2)

pyplot.show()

"""
#########
#ROC curves should be used when there are roughly equal numbers of observations for each class.
#Precision-Recall curves should be used when there is a moderate to large class imbalance.
################ ROC -------------------not working
from sklearn.metrics import precision_recall_curve
#Then we create the ROC Curve with the following code :
y_pred = classifier2.predict_proba(X_t_test)
# keep probabilities for the positive outcome only
y_pred_proba  = y_pred[:, 1]
# calculate scores
ns_probs = [0 for _ in range(len(Y_test))]
ns_auc = roc_auc_score(Y_test, ns_probs)
lr_auc = roc_auc_score(Y_test, y_pred_proba )
# summarize scores
print('Using Yes/No: ROC AUC=%.3f' % (ns_auc))
print('Using probability: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(Y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(Y_test, y_pred_proba)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--') ####### if error use plot no skill
#pyplot.plot([0, 1], [0, 1], linestyle='--') # plot no skill
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='ROC')
######### precesion recall curve
lr_precision, lr_recall, _ = precision_recall_curve(Y_test, y_pred_proba)
pyplot.plot(lr_recall, lr_precision, marker='.', label='Precision-Recall')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


################  XGBoost classifier ############# 

################ XGBoost classifier 2 with bayesian tuning ############# 
'''
To do the bayesian parameter tuning, 
I use the BayesSearchCV class of scikit-optimize. 
It works basically as a drop-in replacement for GridSearchCV and RandomSearchCV,
 but generally I get better results with it. 
 In the following I define the BayesSearchCV object, 
 and write a short convenience function that will be used during 
 optimization to output current status of the tuning. 
 Locally you mayrun with n_jobs=4 for the classifier,
 and n_jobs=6 for the BayesSearchCV object.
'''
my_timer = Timer()
# Classifier
bayes_cv_tuner = BayesSearchCV(
    estimator = xgb.XGBClassifier(
        n_jobs = 1,
        objective = 'binary:logistic',
        eval_metric = 'auc',
        silent=1,
        tree_method='approx'
    ),
    search_spaces = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'min_child_weight': (0, 10),
        'max_depth': (0, 50), 'max_delta_step': (0, 20),
        'subsample': (0.01, 1.0, 'uniform'),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'colsample_bylevel': (0.01, 1.0, 'uniform'),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'gamma': (1e-9, 0.5, 'log-uniform'),
        'n_estimators': (50, 100),
        'scale_pos_weight': (1e-6, 500, 'log-uniform')
    },    
    scoring = 'roc_auc',
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    ),
    n_jobs = 3,
    n_iter = 10,   
    verbose = 0,
    refit = True,
    random_state = 42)
    
def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""
    
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
    
    # Get current parameters and the best parameters    
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))
    
    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name+"_cv_results.csv")

h = bayes_cv_tuner.fit(X_t_train, Y_train, callback=status_print)
######## Time
elapsed = my_timer.get_time()
print("\nTotal compute time was: %s" % elapsed)



########### Learning curve   -------------not working
y_train = np.reshape(bayes_cv_tuner, len(Y_train))
cv_ = ShuffleSplit(n_splits = 20, test_size = 0.20, random_state = 0)
Plot_learning_curve(bayes_cv_tuner, 'Learning Curves', X_t_train, y_train, cv = cv_, n_jobs = 1)
###
#########classification report
y_pred = bayes_cv_tuner.predict(X_t_train)
c_report = classification_report(Y_train, y_pred)
print('\nClassification report:\n', c_report)

########### Confiusion matrix
y_predicted_test = bayes_cv_tuner.predict(X_t_test)
cm = confusion_matrix(Y_test, y_predicted_test)
AUC = roc_auc_score(Y_test, y_predicted_test)
Print_confusion_matrix(cm, AUC, 'Confusion matrics of the test dataset')
##

