import scipy.io as io
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

mat_train = io.loadmat('./data/spamTrain.mat')
X_train = mat_train['X']
y_train = mat_train['y'].ravel()

mat_test = io.loadmat('./data/spamTest.mat')
X_test = mat_test['Xtest']
y_test = mat_test['ytest']

svc = SVC()
svc.fit(X_train,y_train)
score = svc.score(X_test,y_test)
pre = svc.predict(X_test)
print("SVM===============")
print(metrics.classification_report(y_test,pre))



logit = LogisticRegression()
logit.fit(X_train, y_train)
score = logit.score(X_test,y_test)
pre = logit.predict(X_test)
print("LogisticRegression===============")
print(metrics.classification_report(y_test,pre))

print(score)