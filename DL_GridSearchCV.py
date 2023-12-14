import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout
from scikeras.wrappers import KerasRegressor

# Load the CSV data
data = pd.read_csv('allRecipes_users_data.csv', sep=';')

# Separate features (username, recipe_id) and target (rating)
X = data[['username', 'recipe_id']]
y = data['rating']

# Convert 'recipe_id' and 'username' to one-hot encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[['recipe_id', 'username']])
X_encoded = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(['recipe_id', 'username']))
X = pd.concat([X_encoded, X.drop(['recipe_id', 'username'], axis=1)], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the base model function
def create_model(hidden_layer_sizes=(64,), activation='relu', optimizer='adam', dropout_rate=0.0):
    model = Sequential()
    
    # Add embeddings for 'recipe_id' and 'username'
    model.add(Embedding(input_dim=len(X_encoded.columns), output_dim=8, input_length=X_train_scaled.shape[1]))
    model.add(Flatten())
    
    # Add additional hidden layers if applicable
    for units in hidden_layer_sizes:
        model.add(Dense(units, activation=activation))
        model.add(Dropout(dropout_rate))  # Add dropout for regularization
    
    model.add(Dense(1))  # Output layer for regression task
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

# Wrap the Keras model with the KerasRegressor
keras_regressor = KerasRegressor(model=create_model, verbose=1, hidden_layer_sizes=(64,), activation='relu', optimizer='adam', dropout_rate=0.0)

# Define the hyperparameter grid
param_grid = {
    'hidden_layer_sizes': [(64,), (128, 64), (256, 128, 64)],
    'activation': ['relu', 'tanh', 'sigmoid'],
    'batch_size': [32, 64, 128],
    'epochs': [10, 20, 30],
    'optimizer': ['adam', 'sgd', 'rmsprop'],
    'dropout_rate': [0.0, 0.2, 0.5],
}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=keras_regressor, param_grid=param_grid, cv=3, verbose=2, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the model to the data
grid_search_result = grid_search.fit(X_train_scaled, y_train)


# Display the best parameters and results
print("Best Parameters: ", grid_search_result.best_params_)
print("Best Negative Mean Squared Error: ", grid_search.best_score_)


# Get the best model from the grid search
best_model = grid_search_result.best_estimator_
# Evaluate the best model on the test set
test_loss = best_model.score(X_test_scaled, y_test)
print(f"Test Loss: {test_loss}")

#Best Parameters:  {'activation': 'relu', 'batch_size': 16, 'dropout_rate': 0.2, 'epochs': 10, 'hidden_layer_sizes': (64,), 'optimizer': 'adam'}