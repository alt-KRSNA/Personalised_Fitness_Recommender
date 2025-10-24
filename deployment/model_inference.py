# deployment/model_inference.py
import joblib
import os
import numpy as np
from src.config import MODELS_DIR
from src.recommendation import get_plan_for_cluster
from src.utils import load_pipeline, load_classifier, ensure_array

PREPROCESSOR_NAME = "preprocess_pipeline.joblib"
CLASSIFIER_NAME = "classification_model.joblib"

class RecommenderService:
    def __init__(self):
        self.preproc = load_pipeline(PREPROCESSOR_NAME)
        self.clf = load_classifier(CLASSIFIER_NAME)

    def predict_cluster(self, user_dict):
        """
        user_dict: dict with keys matching NUMERIC_FEATURES + CATEGORICAL_FEATURES
        returns: cluster_id, plan
        """
        # keep the same ordering as preprocessing
        feature_order = ["age","height_cm","weight_kg","bmi","sleep_hours","weekly_exercise_minutes",
                         "gender","goal","activity_level","dietary_pref"]
        X = [user_dict.get(k) for k in feature_order]
        X_arr = ensure_array(X)
        X_transformed = self.preproc.transform(X_arr)
        cluster = self.clf.predict(X_transformed)[0]
        plan = get_plan_for_cluster(cluster)
        return int(cluster), plan

# quick test
if __name__ == "__main__":
    svc = RecommenderService()
    sample = {
        "age": 28, "height_cm":170, "weight_kg":70, "bmi":24.2,
        "sleep_hours":7.0, "weekly_exercise_minutes":120,
        "gender":"male","goal":"muscle_gain","activity_level":"moderate","dietary_pref":"high_protein"
    }
    print(svc.predict_cluster(sample))
