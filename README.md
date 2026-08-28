# Dimension Predictor V2

Projet NLP français de classification multi-dimensions.

## Objectif

Entrée :

    morceau de texte

Le modèle analyse 12 dimensions.

Pour chaque dimension :

    1. présence / absence
    2. statut si la dimension est présente

Les dimensions appartiennent à deux groupes :

    MUST
    SHOULD

Les statuts possibles dépendent du groupe.

Pour certaines dimensions MUST, les statuts autorisés
dépendent également de la Source.

## Architecture

    Texte
      |
      v
    Transformer
      |
      +---- présence dimension 1
      |           |
      |           +---- statut
      |
      +---- présence dimension 2
      |           |
      |           +---- statut
      |
      ...
      |
      +---- présence dimension 12
                  |
                  +---- statut

## Transfer learning

Trois modes sont disponibles :

    frozen
    full
    lora

## Installation

Créer un environnement :

    python -m venv .venv

Windows :

    .venv\Scripts\activate

Linux/macOS :

    source .venv/bin/activate

Installer :

    pip install -r requirements.txt

## Workflow

1. Ajouter des textes :

    python scripts/add_text.py

2. Annoter :

    python scripts/annotate.py

3. Vérifier le corpus :

    python scripts/validate_corpus.py

4. Préparer les données :

    python scripts/prepare_data.py

5. Entraîner :

    python scripts/train.py

6. Prédire :

    python scripts/predict.py --text "Texte à analyser" --source entreprise

## Corpus

Le corpus principal est :

    data/annotations/corpus.csv

Il est créé automatiquement par le générateur.
