import pandas as pd

from surprise import SVD, NMF, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, GridSearchCV

userdata = pd.read_csv('allRecipes_users_data.csv', sep = ';')
userdata = userdata.drop('comment', axis = 1)

reader = Reader()

data = Dataset.load_from_df(userdata, reader)

train_set, test_set = train_test_split(data, test_size=0.2)

models = [SVD(), NMF()]

for model in models:
    model.fit(train_set)
    predictions = model.test(test_set)
    print('model: ', model)
    accuracy.rmse(predictions)

# GridSearchCV - - -

# SVD

param_grid_SVD = {
    "n_epochs": [5, 10, 15, 20],
    "lr_all": [0.002, 0.005, 0.01, 0.02],
    "reg_all": [0.2, 0.4, 0.6, 0.8],
    "n_factors": [5, 10, 15],
    "biased": [True, False]
}

gs = GridSearchCV(SVD, param_grid_SVD, measures=['rmse'], cv=5)

gs.fit(data)

print('For SVD:')
print('Best RMSE score: ', gs.best_score['rmse'])
print('Best parameters: ', gs.best_params['rmse'])

# NMF

param_grid_NMF = {
    'n_factors': [5, 10, 15],
    'n_epochs': [10, 20, 30],
    'lr_bu': [0.002, 0.005, 0.01],
    'lr_bi': [0.002, 0.005, 0.01],
    'reg_pu': [0.02, 0.04, 0.06],
    'reg_qi': [0.02, 0.04, 0.06]
}

gs = GridSearchCV(NMF, param_grid_NMF, measures=['rmse'], cv=5)

gs.fit(data)

print('For NMF:')
print('Best RMSE score: ', gs.best_score['rmse'])
print('Best parameters: ', gs.best_params['rmse'])

# Meilleurs paramètres : 'n_epochs': 15, 'lr_all': 0.01, 'reg_all': 0.2, 'n_factors': 5, 'biased': True

svd_model = SVD(n_epochs=15, lr_all=0.01, reg_all=0.2, n_factors=5, biased=True)