import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout

userdata = pd.read_csv('allRecipes_users_data.csv', sep=';')
userdata = userdata.drop('comment', axis = 1)
userdata = userdata[['username', 'recipe_id', 'rating']]
print(userdata.tail())
user_dict = {}

for index, row in userdata.iterrows():
    user_id = row['recipe_id']
    username = row['username']
    
    if user_id not in user_dict:
        user_dict[user_id] = [username]
    else:
        user_dict[user_id].append(username)

user_mapping = {user: i for i, user in enumerate(userdata['username'].unique())}
recipe_mapping = {recipe: i for i, recipe in enumerate(userdata['recipe_id'].unique())}

userdata['user_id'] = userdata['username'].map(user_mapping)
userdata['recipe_id'] = userdata['recipe_id'].map(recipe_mapping)

train_data, test_data = train_test_split(userdata, test_size=0.3, random_state=42)

model = Sequential()

model.add(Embedding(input_dim=len(userdata['user_id'].unique()), output_dim=50, input_length=1, name='user_embedding'))
model.add(Embedding(input_dim=len(userdata['recipe_id'].unique()), output_dim=50, input_length=1, name='recipe_embedding'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='tanh'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='linear'))

model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mae'])

train_inputs = {'user_embedding_input': train_data['user_id'], 'recipe_embedding_input': train_data['recipe_id']}
test_inputs = {'user_embedding_input': test_data['user_id'], 'recipe_embedding_input': test_data['recipe_id']}

model.fit(train_inputs, train_data['rating'], epochs=5, batch_size=64, validation_split=0.1)

test_loss = model.evaluate(test_inputs, test_data['rating'])
print(f'Test Loss: {test_loss}')

'''
new_user_id = user_mapping['daihpos']
new_recipe_id = recipe_mapping['9b5dc451-896e-4105-a750-a9fd11cabee2']
new_input = {'user_embedding_input': np.array([new_user_id]), 'recipe_embedding_input': np.array([new_recipe_id])}

prediction = model.predict(new_input)

print(f'Prediction for user {new_user_id} and recipe {new_recipe_id}: {prediction[0][0]}')

'''
# Fonction de recommandation - - - - - - - - - - - -


recipesdata = pd.read_csv('allRecipes_diabetes_recipes.csv', sep = ';')


def get_recipes_ranking(recipesdata):

    recipes_id = recipesdata['recipe_id']
    ratings = recipesdata['rating']
    rating_counts = recipesdata['rating_count']
    
    recipesranking = []
    
    for i in range (len(recipes_id)):
        
        recipesranking.append([recipes_id[i], ratings[i], rating_counts[i]])

    rating_weight = 0.6
    rating_count_weight = 0.4
    
    for i in range (len(recipesranking)):
        rating = recipesranking[i][1]
        rating_count = recipesranking[i][2]
        score = (rating_weight * rating) + (rating_count_weight * (rating_count / (rating_count + 1)))
        recipesranking[i].append(score)

    recipesranking = sorted(recipesranking, key=lambda x: x[3], reverse = True)
    
    return [{"recipe_id": element[0], "score": element[3]} for element in recipesranking]

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
    
    if username not in userdata['username'].values:
        top_n_recommendations = get_recipes_ranking(recipesdata)[:n]
    
    else:
        recipes_not_rated_by_user = []    

        for recipe_id in user_dict:
            if username not in user_dict[recipe_id]:
                recipes_not_rated_by_user.append(recipe_id) 

        predictions = []
        
        for recipe_id in recipes_not_rated_by_user:
            new_user_id = user_mapping[username]
            new_recipe_id = recipe_mapping[recipe_id]
            new_input = {'user_embedding_input': np.array([new_user_id]), 'recipe_embedding_input': np.array([new_recipe_id])}
            prediction = model.predict(new_input)[0][0]
            predictions.append({"recipe_id": recipe_id, "prediction": prediction})


        predictions.sort(key=lambda x: x.get("prediction"), reverse=True)
        
        top_n_recommendations = predictions[:n]

    for prediction in top_n_recommendations:
        recipe_id = prediction.get("recipe_id","")
        pred = prediction.get("prediction", None)
        if pred is None:
            pred = prediction.get("score", None)
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
            
print(recommandation(username = "arthur"))