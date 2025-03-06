#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <regex>
#include <random>
#include <set>

// Structure pour stocker nos données
struct Dataset {
    std::vector<std::string> text;
    std::vector<int> label;
};

// Classe pour la vectorisation du texte
class CountVectorizer {
private:
    std::set<std::string> stopwords;
    std::map<std::string, int> vocabulary;
    int max_features;

public:
    CountVectorizer(const std::set<std::string>& stop_words, int max_feat = 100)
        : stopwords(stop_words), max_features(max_feat) {}

    std::vector<std::vector<int>> fit_transform(const std::vector<std::string>& texts) {
        // Création du vocabulaire
        std::map<std::string, int> word_freq;

        for (const auto& text : texts) {
            std::istringstream iss(text);
            std::string word;
            while (iss >> word) {
                if (stopwords.find(word) == stopwords.end()) {
                    word_freq[word]++;
                }
            }
        }

        // Sélection des max_features mots les plus fréquents
        //souvent notée X, avec des dimensions n × p, où n est le nombre d'exemples et p le nombre de features.
        std::vector<std::pair<std::string, int>> word_freq_vec(word_freq.begin(), word_freq.end());
        std::sort(word_freq_vec.begin(), word_freq_vec.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        int vocab_size = std::min(max_features, static_cast<int>(word_freq_vec.size()));
        for (int i = 0; i < vocab_size; i++) {
            vocabulary[word_freq_vec[i].first] = i;
        }

        // Création de la matrice de features
        std::vector<std::vector<int>> features(texts.size(), std::vector<int>(vocab_size, 0));
        for (size_t i = 0; i < texts.size(); i++) {
            std::istringstream iss(texts[i]);
            std::string word;
            while (iss >> word) {
                if (vocabulary.find(word) != vocabulary.end()) {
                    features[i][vocabulary[word]]++;
                }
            }
        }

        return features;
    }

    std::vector<std::vector<int>> transform(const std::vector<std::string>& texts) {
        std::vector<std::vector<int>> features(texts.size(), std::vector<int>(vocabulary.size(), 0));
        for (size_t i = 0; i < texts.size(); i++) {
            std::istringstream iss(texts[i]);
            std::string word;
            while (iss >> word) {
                if (vocabulary.find(word) != vocabulary.end()) {
                    features[i][vocabulary[word]]++;
                }
            }
        }
        return features;
    }
};

// Classe pour la régression logistique tres utile pour la classification binaire sur l'apprentissage
class LogisticRegression {
private:
    std::vector<double> weights;
    double learning_rate = 0.01;
    int max_iter = 1000;

public:
    void fit(const std::vector<std::vector<int>>& X, const std::vector<int>& y) {
        weights.resize(X[0].size(), 0.0);

        for (int iter = 0; iter < max_iter; iter++) {
            for (size_t i = 0; i < X.size(); i++) {
                double pred = predict_proba(X[i]);
                double error = y[i] - pred;

                for (size_t j = 0; j < weights.size(); j++) {
                    weights[j] += learning_rate * error * X[i][j];
                }
            }
        }
    }

    std::vector<int> predict(const std::vector<std::vector<int>>& X) {
        std::vector<int> predictions;
        for (const auto& x : X) {
            predictions.push_back(predict_proba(x) > 0.5 ? 1 : 0);
        }
        return predictions;
    }

private:
    double predict_proba(const std::vector<int>& x) {
        double z = 0.0;
        for (size_t i = 0; i < weights.size(); i++) {
            z += weights[i] * x[i];
        }
        return 1.0 / (1.0 + std::exp(-z));
    }
};

// Fonction de nettoyage du texte
std::string clean_text(std::string text) {
    // Conversion en minuscules
    std::transform(text.begin(), text.end(), text.begin(), ::tolower);

    // Suppression des caractères spéciaux qui servent a rien
    std::regex special_chars("[^a-zàáâãäçèéêëìíîïñòóôõöùúûüýÿ0-9\\s]");
    text = std::regex_replace(text, special_chars, "");

    return text;
}

int main() {
    // dataset pour nourrir le llm
    Dataset data;
    data.text = {
        "Je te déteste, tu es horrible !",
        "J'aime beaucoup cette vidéo, merci.",
        "Va te faire voir, imbécile.",
        "Quel contenu inspirant, bravo à l'équipe !",
        "Tu es vraiment nul et inutile.",
        "Je suis impressionné par la qualité de cette vidéo.",
        "Ferme-la, personne ne veut entendre ça.",
        "C'est une discussion constructive, merci pour vos efforts.",
        "Ce commentaire est complètement stupide et inutile.",
        "Merci pour cette vidéo, elle m'a beaucoup aidé !",
        "Personne n'a besoin de voir des bêtises pareilles.",
        "Excellent contenu, continuez comme ça !",
        "Tu ne comprends rien, arrête de commenter.",
        "Bravo, c'est exactement ce que je cherchais.",
        "Espèce d'idiot, tu ne sais même pas de quoi tu parles.",
        "Cette vidéo est très claire, merci pour le travail.",
        "Tu es une honte, personne ne veut lire ça.",
        "Le tutoriel est super bien expliqué, merci !",
        "C'est complètement débile, arrête de poster.",
        "J'adore cette chaîne, toujours des vidéos intéressantes.",
        "Dégage d'ici, personne ne te supporte.",
        "Merci pour ces conseils, c'est vraiment utile.",
        "T'es vraiment le pire, tes vidéos sont nulles.",
        "Une très bonne vidéo, claire et précise, bravo !",
        "cest un connard il est mechant",
        "connard de merde"
        // ... ajoutez le reste des textes
    };
    data.label = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,1,1};

    // Nettoyage des textes
    std::vector<std::string> cleaned_texts;
    for (const auto& text : data.text) {
        cleaned_texts.push_back(clean_text(text));
    }

    // Définition des stop words
    std::set<std::string> french_stopwords = {
        "le", "la", "les", "un", "une", "des", "du", "de", "dans",
        // ... ajoutez le reste des stop words
    };

    // Vectorisation
    CountVectorizer vectorizer(french_stopwords, 100);
    auto X = vectorizer.fit_transform(cleaned_texts);
    auto y = data.label;

    // Division des données (train/test)
    // Implémentez votre propre fonction de division ou utilisez une bibliothèque

    // Entraînement du modèle
    LogisticRegression model;
    model.fit(X, y);

    // test sur de nouveaux commentaires
    std::vector<std::string> new_comments = {
        "Je ne supporte pas cette personne.",
        "Cette vidéo est incroyable, merci pour votre travail.",
        "Arrête de dire n'importe quoi, imbécile.",
        "Une excellente présentation, bravo à toute l'équipe."
    ,"connard vas travailler"
    };

    std::vector<std::string> new_comments_clean;
    for (const auto& comment : new_comments) {
        new_comments_clean.push_back(clean_text(comment));
    }

    auto new_comments_vectorized = vectorizer.transform(new_comments_clean);
    auto predictions = model.predict(new_comments_vectorized);

    for (size_t i = 0; i < new_comments.size(); i++) {
        std::cout << "Commentaire : '" << new_comments[i]
                  << "' -> " << (predictions[i] == 1 ? "Haineux" : "Non haineux") << std::endl;
    }

    return 0;
}
