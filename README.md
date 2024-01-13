# Projet Master - Système d'Assistance Alimentaire pour Régimes Spéciaux - Partie Modèle

De nos jours, beaucoup de personnes ont des restrictions alimentaires, ou
souhaitent manger plus léger ou moins sucré dans le but de perdre du poids. Trouver
la recette adaptée à son régime peut s’avérer difficile : il existe de nombreux sites
répertoriant plusieurs recettes mais en trouver une nous correspondant passe par une
recherche détaillée et pouvant être complexe. C’est dans ce contexte que nous nous
plaçons : on sait que l’intelligence artificielle peut apprendre de données pour prédire
une valeur cible. On arrive donc à la problématique suivante : Comment concevoir et
développer un système de recommandation de recettes de cuisine intelligent qui
prend en compte les régimes spéciaux et propose des recettes adaptées aux besoins
nutritionnels, aux allergies alimentaires et aux préférences alimentaires des
utilisateurs ?

L’idée de base est de créer une fonction de recommandation utilisant des modèles de
Deep Learning pour fournir à un utilisateur la recette qui lui correspond le mieux.
Notre objectif final est de concevoir une application mobile dans laquelle est intégrée
cette fonction de recommandation.

## Collecte de données

Après avoir exploré les offres existantes sur le marché, nous avons entrepris
une phase approfondie de recherche. Notre objectif était de recueillir des données à
partir de nombreux sites web, comprenant une variété de recettes ainsi que des avis
d'utilisateurs. Cette étape était cruciale pour alimenter les informations nécessaires à
la construction de notre futur modèle. Une fois que nous avons établi une liste de
sites pertinents, nous avons développé des scripts Python dédiés au webscraping en
utilisant la bibliothèque Selenium. L'avantage notable de cette bibliothèque réside
dans sa capacité à traiter non seulement les sites statiques, mais également les sites
dynamiques, contrairement au package BeautifulSoup.

Pour chaque catégorie de problèmes alimentaires, nous générions deux fichiers CSV
distincts. Le premier fichier est conçu pour stocker les informations relatives aux
recettes, comprenant leur identifiant, leur nom, leur lien internet, leurs ingrédients,
leurs valeurs nutritionnelles, leur note globale, ainsi que le nombre d'évaluations
reçues. Le deuxième fichier enregistre les commentaires et les notes attribués par
chaque utilisateur. L'objectif est d'identifier les utilisateurs ayant évalué plusieurs
recettes différentes, car notre modèle sera fondé sur ces relations.

## Traitement des données

La phase finale du processus de collecte consiste à traiter les données
obtenues. En examinant attentivement nos tableurs, nous avons identifié plusieurs
éléments à rectifier, tels que des caractères spéciaux, des unités (mg, g) présentes
dans les colonnes des apports nutritionnels, et la présence d'utilisateurs anonymes, ce
qui compromet la mise en relation des recettes. Pour remédier à ces problèmes, nous
avons élaboré un script Python visant à nettoyer nos fichiers CSV, rendant ainsi les
données exploitables.

Le plus important concernait le travail sur la colonne des noms d’utilisateurs.
En effet, bien que l’on ait généré des id uuid pour les recettes, concernant les
utilisateurs, nous avons fait le choix d’utiliser leur nom comme identifiant. Cela
implique deux choses : chaque nom d’utilisateur doit être unique et ne doit contenir
aucun caractère spécial ou majuscule. Grâce à notre traitement, nous pouvons donner
à nos modèles des entrées convenables.

## Choix du modèle et de ses hyperparamètres

Nous utilisons la bibliothèque Keras de Tensorflow pour
concevoir notre propre modèle d’apprentissage en définissant nous-même ses
couches de neurones. Keras est une interface de haut niveau écrite en Python, qui
permet de créer, entraîner et évaluer des modèles d'apprentissage profond de manière
simple et efficace. Intégré à TensorFlow, Keras offre une abstraction conviviale tout
en conservant une grande flexibilité. Concernant le modèle, nous l’avons construit
avec Sequential() et il contient :

- Une couche Embedding : Le modèle commence par une couche
d'incorporation qui prend en compte deux variables catégorielles, 'recipe_id' et
'username'. Cette couche crée des représentations denses pour ces catégories,
permettant au réseau de capturer des relations complexes entre les recettes et
les utilisateurs.

-  Une couche Flatten : Ensuite, une couche Flatten est ajoutée pour aplatir les
sorties de la couche d'incorporation en un vecteur unidimensionnel. Cela
permet de convertir les informations des catégories en une forme compatible
avec les couches entièrement connectées qui suivent.

- Couches Cachées : Le modèle prend en charge la spécification de couches
cachées additionnelles et de leur taille via le paramètre hidden_layer_sizes.
Chaque couche cachée est une couche dense (Dense) avec une fonction
d'activation relu et un dropout associé pour la régularisation. Cette architecture
flexible permet d'ajuster la complexité du modèle en fonction des données
spécifiques.

- Couche de Sortie : Enfin, une couche dense avec une seule unité est ajoutée
en tant que couche de sortie pour la tâche de régression. La fonction
d'activation par défaut, linéaire, est utilisée pour obtenir une valeur continue en
sortie.

- Compilation : Le modèle est compilé en utilisant la fonction de perte
'mean_squared_error' (erreur quadratique moyenne) et l'optimiseur spécifié par
le paramètre optimizer.

Pour tirer le meilleur de chaque modèle, nous avons utilisé la librairie
GridSearchCV: l'optimisation des hyperparamètres est une étape cruciale dans le
processus de construction de modèles d'apprentissage automatique afin d'améliorer
leurs performances. Dans ce contexte, l'utilisation de GridSearchCV, une technique
de recherche systématique des meilleurs hyperparamètres, s'avère extrêmement utile.
14

GridSearchCV, fourni par la bibliothèque scikit-learn en Python, effectue une
recherche exhaustive dans l'espace des hyperparamètres spécifié, évaluant chaque
combinaison possible à l'aide d'une validation croisée. Cette méthode permet
d'identifier les hyperparamètres qui maximisent les performances du modèle sur les
données d'entraînement tout en évitant le surajustement. En spécifiant une grille
d'hyperparamètres à explorer, les utilisateurs peuvent parcourir différentes
combinaisons de valeurs et obtenir les paramètres optimaux pour leur modèle. Bien
que cette approche soit gourmande en termes de calcul, elle offre une solution
robuste pour trouver les meilleures configurations d'hyperparamètres, améliorant
ainsi la généralisation du modèle sur des données inconnues. L'intégration de
GridSearchCV dans le processus de développement de modèles permet d'optimiser
efficacement les performances et de garantir une plus grande fiabilité dans les
résultats obtenus. Voici les grilles de paramètres que nous avons fourni à
GridSearchCV pour chaque modèle ainsi que les meilleurs paramètres trouvés.

