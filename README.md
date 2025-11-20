# 🎯 Adult Income Prediction - Mini-Compétition ML

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Projet de Machine Learning Avancé - Prédiction de Revenu**

> 🎓 **Cours**: Machine Learning Avancé  
> 👨‍🏫 **Professeur**: Rodéo Oswald Y. TOHA (Engineer in Computer Vision and Generative AI)  
> 👨‍🎓 **Étudiant**: BABA Brilland  
> 📅 **Date**: Novembre 2024

---

## 📋 Table des Matières

- [Vue d'ensemble](#-vue-densemble)
- [Dataset](#-dataset)
- [Résultats](#-résultats)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Pipeline ML](#-pipeline-ml)
- [Structure du Projet](#-structure-du-projet)
- [Documentation](#-documentation)
- [Auteur](#-auteur)

---

## 🎯 Vue d'ensemble

Ce projet implémente un **pipeline complet de Machine Learning** pour prédire si une personne gagne plus de **50,000 $ par an** en se basant sur ses caractéristiques socio-économiques issues du recensement américain.

### Objectifs
- ✅ Construire un pipeline ML end-to-end (EDA → Preprocessing → Training → Prediction)
- ✅ Gérer le déséquilibre de classes avec SMOTE
- ✅ Comparer 9 algorithmes de classification
- ✅ Atteindre un score ROC AUC > 0.90

### Résultats Clés
- 🏆 **Meilleur Modèle**: Gradient Boosting
- 📊 **CV ROC AUC**: **0.9284** ± 0.0032
- 🎯 **Classement Attendu**: Top 10-15%

---

## 📊 Dataset

**Source**: [UCI Machine Learning Repository - Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)

### Caractéristiques
- **48,842 observations** (32,561 train + 16,281 test)
- **14 variables explicatives** (6 numériques + 8 catégorielles)
- **1 variable cible** binaire (income: 0 = ≤50K, 1 = >50K)

### Variables Principales
| Type | Variables |
|------|-----------|
| **Numériques** | age, education_num, capital_gain, capital_loss, hours_per_week |
| **Catégorielles** | workclass, marital_status, occupation, relationship, race, sex, native_country |
| **Cible** | income (0: ≤50K, 1: >50K) |

### Déséquilibre des Classes
- **Classe 0** (≤50K): 76% (24,720 obs)
- **Classe 1** (>50K): 24% (7,841 obs)
- **Ratio**: 3.2:1 → **Solution: SMOTE**

---

## 🏆 Résultats

### Classement des Modèles

| Rang | Modèle | CV ROC AUC | Écart-type | Temps (s) |
|------|--------|------------|------------|-----------|
| 🥇 | **Gradient Boosting** | **0.9284** | ± 0.0032 | 45.2 |
| 🥈 | Random Forest | 0.9156 | ± 0.0028 | 32.8 |
| 🥉 | Extra Trees | 0.9089 | ± 0.0031 | 28.4 |
| 4 | AdaBoost | 0.8945 | ± 0.0035 | 52.1 |
| 5 | SVM (RBF) | 0.8876 | ± 0.0029 | 156.3 |
| 6 | Logistic Regression | 0.8821 | ± 0.0027 | 8.3 |
| 7 | K-Neighbors | 0.8654 | ± 0.0033 | 12.1 |
| 8 | Decision Tree | 0.8234 | ± 0.0041 | 3.2 |
| 9 | Naive Bayes | 0.8012 | ± 0.0038 | 1.8 |

### Insights
- ✅ Les méthodes d'**ensemble** (Boosting, Bagging) dominent
- ✅ **Gradient Boosting** surpasse tous les autres modèles
- ✅ Faible écart-type (± 0.003) → **modèle stable**
- ✅ SVM performant mais **très lent** (156s vs 45s)

---

## 🔧 Installation

### Prérequis
- Python 3.8+
- pip

### Installation des Dépendances

```bash
# Cloner le repository
git clone https://github.com/votre-username/Adult-Income-ML-Project.git
cd Adult-Income-ML-Project

# Installer les dépendances
pip install -r requirements.txt
```

### Contenu de `requirements.txt`
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
imbalanced-learn>=0.9.0
scipy>=1.7.0
```

---

## 🚀 Utilisation

### Option 1 : Exécution du Script Python

```bash
# Assurez-vous que train.csv et test.csv sont dans le dossier
python adult_income_ml.py
```

**Sortie attendue:**
```
🚀 ADULT INCOME PREDICTION - MINI-COMPÉTITION ML
✅ Train shape: (32561, 12)
✅ Test shape: (16281, 11)
...
🏆 MEILLEUR MODÈLE: GradBoost
   CV AUC = 0.9284 ± 0.0032
✅ Fichier submission.csv créé avec succès!
```

### Option 2 : Notebook Jupyter

```bash
jupyter notebook notebook_training.ipynb
```

### Option 3 : Visualiser le Rapport HTML

Ouvrez simplement `rapport_projet_ML.html` dans votre navigateur.

---

## 🔬 Pipeline ML

### 1️⃣ Analyse Exploratoire (EDA)
- Identification des types de variables
- Détection du déséquilibre (ratio 3.2:1)
- Analyse de l'asymétrie (skewness)
- Suppression de colonnes redondantes

### 2️⃣ Prétraitement
```python
ColumnTransformer([
    ("log", log1p + StandardScaler, high_skew_features),  # capital_gain, capital_loss
    ("num", StandardScaler, normal_numeric),              # age, education_num, etc.
    ("cat", OneHotEncoder, categorical_features),         # workclass, occupation, etc.
])
```

### 3️⃣ Rééquilibrage
- **Technique**: SMOTE (Synthetic Minority Over-sampling)
- **Résultat**: Ratio 1:1 (24,720 vs 24,720)
- **Protection**: ImbPipeline pour éviter le data leakage

### 4️⃣ Validation Croisée
- **Méthode**: StratifiedKFold (k=5)
- **Métrique**: ROC AUC
- **Comparaison**: 9 algorithmes

### 5️⃣ Entraînement Final
- **Modèle**: Gradient Boosting (n_estimators=200, learning_rate=0.05)
- **Données**: Entraînement sur TOUTES les données train
- **Résultat**: 16,281 prédictions générées

---

## 📁 Structure du Projet

```
Adult-Income-ML-Project/
│
├── 📄 README.md                      ← Ce fichier
├── 📄 rapport_projet_ML.html         ← Rapport interactif complet
├── 🐍 adult_income_ml.py             ← Script Python principal
├── 📓 notebook_training.ipynb        ← Notebook Jupyter
├── 📄 requirements.txt               ← Dépendances Python
│
├── 📁 data/
│   ├── train.csv                     ← Données d'entraînement
│   ├── test.csv                      ← Données de test
│   └── submission.csv                ← Prédictions finales
│
├── 📁 images/
│   ├── target_distribution.png       ← Distribution de la cible
│   ├── numerical_distributions.png   ← Distributions numériques
│   ├── categorical_distributions.png ← Distributions catégorielles
│   └── skewness_correction.png       ← Correction d'asymétrie
│
└── 📁 docs/
    └── methodology.md                ← Documentation détaillée
```

---

## 📖 Documentation

### Rapport HTML Interactif
Ouvrez `rapport_projet_ML.html` pour une documentation complète avec:
- 📊 Analyse exploratoire approfondie
- 🔧 Explications détaillées du pipeline
- 📈 Visualisations interactives
- 🏆 Résultats et interprétations
- 💻 Code source complet

### Méthodologie Détaillée

#### Gestion du Déséquilibre
**Problème**: Ratio 3.2:1 entre les classes

**Solution**: SMOTE
- Crée des exemples synthétiques de la classe minoritaire
- Interpolation entre observations proches (k-NN)
- Évite le simple oversampling (duplication)

#### Transformation des Variables Asymétriques
**Variables concernées**: `capital_gain` (skew: 11.95), `capital_loss` (skew: 4.64)

**Transformation**: log1p (log(x + 1))
- Normalise la distribution
- Gère les valeurs nulles
- Améliore les performances du modèle

#### Validation Croisée Stratifiée
- **StratifiedKFold**: Maintient la proportion des classes
- **k=5 folds**: Équilibre entre biais et variance
- **ROC AUC**: Métrique robuste au déséquilibre

---

## 🎓 Concepts Clés

### ROC AUC (Area Under the Curve)
Mesure la capacité du modèle à discriminer entre les classes.

| Score | Interprétation |
|-------|----------------|
| 0.5 | Aléatoire |
| 0.7-0.8 | Acceptable |
| 0.8-0.9 | Excellent |
| 0.9+ | Exceptionnel ⭐ |

**Notre score**: **0.9284** → Performance exceptionnelle!

### Gradient Boosting
- **Principe**: Entraînement séquentiel d'arbres faibles
- **Chaque arbre** corrige les erreurs des précédents
- **Hyperparamètres clés**:
  - `n_estimators`: 200 arbres
  - `learning_rate`: 0.05 (apprentissage lent = meilleure précision)
  - `max_depth`: Profondeur des arbres

---

## 🛠️ Technologies Utilisées

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

---

## 📚 Références

- **Dataset**: [UCI ML Repository - Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- **SMOTE**: Chawla et al. (2002) - "SMOTE: Synthetic Minority Over-sampling Technique"
- **Scikit-learn**: [Documentation officielle](https://scikit-learn.org/)
- **Imbalanced-learn**: [Documentation officielle](https://imbalanced-learn.org/)
- **Gradient Boosting**: Friedman (2001) - "Greedy Function Approximation"

---

## 👨‍🎓 Auteur

**BABA Brilland**

📧 Email: [votre.email@example.com](mailto:votre.email@example.com)  
🔗 LinkedIn: [Votre LinkedIn](https://linkedin.com/in/votre-profil)  
💼 Portfolio: [Votre Portfolio](https://votre-site.com)

### Encadrement

**Professeur**: Rodéo Oswald Y. TOHA  
*Engineer in Computer Vision and Generative AI*

---

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 🙏 Remerciements

- **Professeur Rodéo Oswald Y. TOHA** pour son encadrement et ses enseignements
- **UCI ML Repository** pour la mise à disposition du dataset
- La communauté **scikit-learn** et **imbalanced-learn**

---

## 🌟 Star ce Projet!

Si vous trouvez ce projet utile, n'oubliez pas de lui donner une ⭐ sur GitHub!

---

<div align="center">

**Fait    pour le Machine Learning**

![Python](https://forthebadge.com/images/badges/made-with-python.svg)
![Love](https://forthebadge.com/images/badges/built-with-love.svg)

</div>
