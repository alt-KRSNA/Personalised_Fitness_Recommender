## # Personalized Fitness Recommender

## Overview
Generates fake user data (5000 entries), clusters users and trains a classifier to predict cluster for new users. Exposes a Flask app to get personalized fitness recommendations.

## Setup
1. Create virtualenv and install:

2. Train models (generates data, preprocess, cluster, classifier):

This creates:
- `data/raw/fake_users.csv`
- preprocessor and models in `models/`

3. Run the Flask app:
Open http://127.0.0.1:5000

## Files
- `src/` : data generation, preprocessing, modeling
- `deployment/` : Flask app + inference
- `models/` : saved models after training

## Notes
- This is a template. Improve recommendations by replacing rule-based mapping with data-driven plans.
