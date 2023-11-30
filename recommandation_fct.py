import pandas as pd

from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

userdata = pd.read_csv('allRecipes_users_data.csv', sep = ';')
userdata = userdata.drop('comment', axis = 1)
userdata = userdata[['username', 'recipe_id', 'rating']]
recipesdata = pd.read_csv('allRecipes_diabetes_recipes.csv', sep = ';')
user_dict = {}

for index, row in userdata.iterrows():
    user_id = row['recipe_id']
    username = row['username']
    
    if user_id not in user_dict:
        user_dict[user_id] = [username]
    else:
        user_dict[user_id].append(username)

reader = Reader()

data = Dataset.load_from_df(userdata, reader)

train_set, test_set = train_test_split(data, test_size=0.2)


svd_model = SVD(n_epochs=15, lr_all=0.01, reg_all=0.2, n_factors=5, biased=True)

svd_model.fit(train_set)

def nutrition_filter(cal_limit, fat_limit, sodium_limit, sugars_limit, protein_limit, cal, fat, sodium, sugars, protein):
    
    if cal_limit is not None:
        if cal > cal_limit:
            return True
    
    if fat_limit is not None:
        if fat > fat_limit:
            return True
    
    if sodium_limit is not None:
        if sodium > sodium_limit:
            return True
    
    if sugars_limit is not None:
        if sugars > sugars_limit:
            return True
    
    if protein_limit is not None:
        if protein > protein_limit:
            return True
    
    return False


def recommandation(username=None, cal_limit = None, fat_limit = None, sodium_limit = None, sugars_limit = None, protein_limit = None, n=10):

    recipes_not_rated_by_user = []    

    for recipe_id in user_dict:
        if username not in user_dict[recipe_id]:
            recipes_not_rated_by_user.append(recipe_id)

    
    predictions = [svd_model.predict(username, recipe_id) for recipe_id in recipes_not_rated_by_user]
    
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    top_n_recommendations = predictions[:n]
    
    for prediction in top_n_recommendations:
        recipe_id = prediction.iid
        pred = prediction.est
        recipedata = recipesdata[recipesdata['recipe_id'] == recipe_id]
        name = recipedata['name'].values[0]
        cal = recipedata['calories'].values[0]
        fat = recipedata['fat'].values[0]
        sodium = recipedata['sodium'].values[0]
        sugars = recipedata['sugars'].values[0]
        protein = recipedata['protein'].values[0]
        
        # Filtre
        
        if not nutrition_filter(cal_limit, fat_limit, sodium_limit, sugars_limit, protein_limit, cal, fat, sodium, sugars, protein):
            print(f"ID: {recipe_id} Estimation: {pred} Name: {name}")
        
recommandation()
