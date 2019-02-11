import scipy.io as io
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


mat = io.loadmat('./data/ex6data3.mat')

data_train = pd.DataFrame(mat.get("X"),columns=['X1_train','X2_train'])
data_train['y_train'] = mat.get("y")

data_val = pd.DataFrame(mat.get("Xval"),columns=["X1_val","X2_val"])
data_val['y_val'] = mat.get("yval")

candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

grid = [{'C':candidate, 'gamma':candidate}]
svc = SVC(kernel='rbf')
grid_search = GridSearchCV(svc,grid)
grid_search.fit(data_train[['X1_train','X2_train']],data_train['y_train'])
print(grid_search.best_params_)
print(grid_search.best_score_)


# grid = [[c,gamma] for c in candidate for gamma in candidate]
#
# final_score = 0
# final_param = []
# best_params = {'C': None, 'gamma': None}
# i = 0
#
# for param in grid:
#     c = param[0]
#     gamma = param[1]
#     svc = SVC(kernel='rbf',gamma=gamma,C=c)
#     svc.fit(data_train[['X1_train','X2_train']],data_train['y_train'])
#     score = svc.score(data_val[['X1_val','X2_val']],data_val['y_val'])
#
#     if score > final_score:
#         final_score = score
#         best_params['C'] = c
#         best_params['gamma'] = gamma
#         print(score)
#         print(c)
#         print(gamma)
#         print("============")
#
# print(best_params)
# print(final_score)




