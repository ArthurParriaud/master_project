import pandas as pd

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
    
    return recipesranking

print(get_recipes_ranking(recipesdata))