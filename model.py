import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout

# Charger les données
userdata = pd.read_csv('allRecipes_users_data.csv', sep=';')
userdata = userdata.drop('comment', axis = 1)
userdata = userdata[['username', 'recipe_id', 'rating']]

# Mapper les utilisateurs et les recettes à des ID numériques
user_mapping = {user: i for i, user in enumerate(userdata['username'].unique())}
recipe_mapping = {recipe: i for i, recipe in enumerate(userdata['recipe_id'].unique())}

userdata['user_id'] = userdata['username'].map(user_mapping)
userdata['recipe_id'] = userdata['recipe_id'].map(recipe_mapping)

# Diviser les données en ensembles d'entraînement et de test
train_data, test_data = train_test_split(userdata, test_size=0.3, random_state=42)

# Créer le modèle
model = Sequential()

model.add(Embedding(input_dim=len(userdata['user_id'].unique()), output_dim=50, input_length=1, name='user_embedding'))
model.add(Embedding(input_dim=len(userdata['recipe_id'].unique()), output_dim=50, input_length=1, name='recipe_embedding'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='tanh'))
model.add(Dropout(0.5))

# Ajouter une couche dense pour la prédiction finale
model.add(Dense(1, activation='linear'))

# Compiler le modèle
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mae'])

# Renommer les clés du dictionnaire de train et de test
train_inputs = {'user_embedding_input': train_data['user_id'], 'recipe_embedding_input': train_data['recipe_id']}
test_inputs = {'user_embedding_input': test_data['user_id'], 'recipe_embedding_input': test_data['recipe_id']}

# Entraîner le modèle
model.fit(train_inputs, train_data['rating'], epochs=5, batch_size=64, validation_split=0.1)

# Évaluer le modèle sur l'ensemble de test
test_loss = model.evaluate(test_inputs, test_data['rating'])
print(f'Test Loss: {test_loss}')