# Détecteur d'Insultes pour Jeux Vidéo

## Présentation
Ce projet implémente un système de détection automatique de contenu haineux ou insultant dans les messages de chat de jeux vidéo. Utilisant des techniques d'apprentissage automatique, il permet de filtrer les messages inappropriés pour maintenir un environnement de jeu plus sain et agréable.

## Fonctionnalités
- Détection de commentaires haineux en français
- Classification binaire (haineux/non haineux)
- Pré-traitement automatique du texte
- Vectorisation basée sur la fréquence des mots
- Filtrage des mots vides (stopwords)
- Modèle de régression logistique pour la classification

## Structure du code
- **Dataset** : Structure de données pour stocker les textes et leurs étiquettes
- **CountVectorizer** : Classe pour transformer le texte en vecteurs numériques
- **LogisticRegression** : Classe pour classifier les messages
- **clean_text** : Fonction de nettoyage du texte (minuscules, suppression des caractères spéciaux)

## Dépendances
- C++ standard (C++11 ou supérieur)
- Bibliothèques standard C++ (iostream, vector, string, map, algorithm, regex, random, set)

## Compilation
```bash
g++ -std=c++11 -o ia main.cpp
```

## Utilisation
1. Exécutez le programme compilé
```bash
./ia
```

2. Le programme analysera les commentaires d'exemple et affichera les résultats de classification

3. Pour utiliser avec vos propres données :
   - Modifiez le vecteur `data.text` pour inclure vos exemples d'entraînement
   - Ajustez le vecteur `data.label` avec les classifications correspondantes (1 pour haineux, 0 pour non haineux)
   - Modifiez le vecteur `new_comments` pour tester votre modèle sur de nouveaux messages

## Personnalisation
- Modifiez `french_stopwords` pour ajouter ou supprimer des mots vides
- Ajustez `max_features` dans CountVectorizer pour modifier la taille du vocabulaire
- Modifiez `learning_rate` et `max_iter` dans LogisticRegression pour ajuster l'apprentissage

## Améliorations possibles
- Ajout de validation croisée pour évaluer la performance du modèle
- Implémentation d'autres algorithmes (SVM, Random Forest, etc.)
- Utilisation de techniques plus avancées (word embeddings, TF-IDF)
- Support multilingue
- Intégration directe avec les API de jeux vidéo

