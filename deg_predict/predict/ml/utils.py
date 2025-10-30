import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import numpy as np
from upload_dataset.models import RiskIndicator
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import os


def predict_with_confidence(model, input_data):
    if hasattr(input_data, 'values'):
        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)
    else:
        prediction = model.predict([input_data])
        proba = model.predict_proba([input_data])

    predicted_value = prediction[0]
    confidence = np.max(proba[0])

    if isinstance(predicted_value, str):
        result = "likely to complete a degree" if predicted_value.lower(
        ) == "completed" else "unlikely to complete degree"
    else:
        result = "likely to complete a degree" if predicted_value == 1 else "unlikely to complete degree"

    return result, confidence


def analyze_sentiment_with_confidence(model, feedback):
    sentiment = model.predict([feedback])[0]
    proba = model.predict_proba([feedback])[0]
    confidence = np.max(proba)

    if sentiment == 1.0:
        return "Negative", confidence
    elif sentiment == 2.0:
        return "Positive", confidence
    else:
        return "Neutral", confidence
    print(confidence)


def calculate_combined_predictions(dt_pred, dt_conf, svm_pred, svm_conf):
    dt_value = 1 if dt_pred.lower().strip() == "likely to complete a degree" else 0
    svm_value = 1 if svm_pred == "Positive" else (
        0.5 if svm_pred == "Neutral" else 0)

    if dt_conf > 0.9:
        weighted_score = dt_value * dt_conf
    else:
        weighted_score = (dt_value * dt_conf * 0.6) + \
            (svm_value * svm_conf * 0.4)

    average_score = (dt_value * dt_conf * 0.5) + (svm_value * svm_conf * 0.5)
##############
   # if weighted_score >= 0.6:
   #     weighted_pred = ("likely to complete a degree", weighted_score)
   # elif weighted_score >= 0.4:
   #     weighted_pred = (
   #         "moderately likely to complete a degree", weighted_score)
   # else:
   #     weighted_pred = ("unlikely to complete degree", weighted_score)

    # if average_score >= 0.6:
    #    average_pred = ("likely to complete a degree", average_score)
    # elif average_score >= 0.4:
    #    average_pred = (
    #       "moderately likely to complete a degree", average_score)
    # else:
    #   average_pred = ("unlikely to complete degree", average_score)

    # return weighted_pred, weighted_score, average_pred, average_score
  #########
    if weighted_score >= 0.6:
        code="Likely to complete a degree"
        if svm_pred == "Positive":
            weighted_pred = (
                "You are likely to complete your degree, and your positive attitude supports your success", weighted_score)
        elif svm_pred == "Negative":
            
            weighted_pred = (
                "You are likely to complete your degree, but consider addressing the concerns reflected in your feedback", weighted_score)
        else:
            
            weighted_pred = (
                "You are likely to complete your degree, even though your attitude appears neutral", weighted_score)
    elif weighted_score >= 0.4:
        code="Moderately likely to complete a degree"
        if svm_pred == "Positive":            
            weighted_pred = (
                "You are moderately likely to complete your degree, and your positive outlook is helpful", weighted_score)
        elif svm_pred == "Negative":            
            weighted_pred = (
                "You are moderately likely to complete your degree, but your negative sentiment may be a risk factor", weighted_score)
        else:            
            weighted_pred = (
                "You are moderately likely to complete your degree, though your neutral feedback suggests uncertainty", weighted_score)
    else:
        code="Unlikely to complete degree"
        if svm_pred == "Positive":
            
            weighted_pred = (
                "You are currently at risk of not completing your degree, but your positive attitude may help you succeed ", weighted_score)
        elif svm_pred == "Negative":
            
            weighted_pred = (
                "There is a risk of not completing your degree. Consider seeking support to address academic and emotional challenges", weighted_score)
        else:            
            weighted_pred = (
                "You may be at risk of not completing your degree. Try to engage more actively in your studies and seek guidance", weighted_score)

    if average_score >= 0.6:
        average_pred = ("Likely to complete a degree", average_score)
    elif average_score >= 0.4:
        average_pred = (
            "Moderately likely to complete a degree", average_score)
    else:
        average_pred = ("Unlikely to complete degree", average_score)

    return weighted_pred, weighted_score, average_pred, average_score, code


def analyze_dropout_factors(input_data):
    risk_factors = []

    if input_data['shs_gpa'].values[0] < -0.5:
        risk_factors.append("Low Senior High GPA")
    if input_data['coll_gpa'].values[0] < -0.5:
        risk_factors.append("Low College GPA")
    if input_data['is_scholar'].values[0] == 0:
        risk_factors.append("Not a Scholar")
    if input_data['is_boarding'].values[0] == 1:
        risk_factors.append("Lives in a Boarding House")
    if input_data['is_living_with_family'].values[0] == 0:
        risk_factors.append("Not Living with Family")
    if input_data['financial_status'].values[0] < -0.5:
        risk_factors.append("Low Parental Income")
    if input_data['father_educ'].values[0] <= 2:
        risk_factors.append("Low Father Education")
    if input_data['mother_educ'].values[0] <= 2:
        risk_factors.append("Low Mother Education")

    return risk_factors


def analyze_dropout_factors1(input_data, stud_id):
    risk_factors = []

    if input_data['shs_gpa'].values[0] > 2.5:
        risk_factors.append("Low Senior High GPA")
    if input_data['coll_gpa'].values[0] > 2.5:
        risk_factors.append("Low College GPA")
    if input_data['is_scholar'].values[0] == 0:
        risk_factors.append("Not a Scholar")
    if input_data['is_boarding'].values[0] == 1:
        risk_factors.append("Lives in a Boarding House")
    if input_data['is_living_with_family'].values[0] == 0:
        risk_factors.append("Not Living with Family")
    if input_data['father_educ'].values[0] <= 3:
        risk_factors.append("Low Father Education")
    if input_data['mother_educ'].values[0] <= 3:
        risk_factors.append("Low Mother Education")
    if input_data['financial_status'].values[0] < 2.5:
        risk_factors.append("Low Parental Income")

    if stud_id:
        for risk in risk_factors:
            RiskIndicator.objects.create(
                student_id=stud_id, risk_message=risk)

    return risk_factors


def train_decision_tree(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    dt_classifier = DecisionTreeClassifier(
        # max_depth=10,
        # min_samples_leaf=2,
        # min_samples_split=4,
        # min_impurity_decrease=0.01,
        # ccp_alpha=0.1,
        # random_state=42
        criterion='entropy',
        max_depth=10,
        min_samples_leaf=2,
        min_samples_split=4,
        min_impurity_decrease=0.01,
        ccp_alpha=0.1,  # Matches RapidMiner's pruning to some extent
        random_state=42
    )

    dt_classifier.fit(X_train, y_train)
    y_pred = dt_classifier.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
        
    print("\nDecision Tree Performance:")
    print("Accuracy Score:", acc)
    print("Precision Score:", prec)
    print("Recall Score:", rec)
    print("F1 Score:", f1)
    
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return dt_classifier, acc, prec, rec, f1

def train_svm_sentiment(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    svm_pipeline = make_pipeline(
        TfidfVectorizer(),
        SVC(probability=True, kernel='rbf', C=1.0, gamma='scale')
    )

    svm_pipeline.fit(X_train, y_train)
    y_pred = svm_pipeline.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)

    joblib.dump(svm_pipeline, os.path.join('predict/ml', 'svm_feedback.pkl'))
    joblib.dump(report, os.path.join('predict/ml', 'svm_metrics.pkl'))

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    return svm_pipeline, acc, prec, rec, f1
# ******************************************
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os

# Define path for saving the model and metrics
MODEL_DIR = os.path.join('predict', 'ml')
os.makedirs(MODEL_DIR, exist_ok=True)


def train_svm_academic(X, y):
    """
    Trains an SVM model using academic features to predict degree completion.
    Saves the model and performance metrics to disk.
    Returns the trained model and performance scores.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    svm_model = make_pipeline(
        StandardScaler(),
        SVC(probability=True, kernel='rbf', C=1.0, gamma='scale')
    )

    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save model and metrics
    joblib.dump(svm_model, os.path.join(MODEL_DIR, 'svm_academic.pkl'))
    joblib.dump(report, os.path.join(MODEL_DIR, 'svm_academic_metrics.pkl'))

    print("\n✅ SVM Academic Model Trained:")
    print(f"Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")

    return svm_model, acc, prec, rec, f1


def predict_svm_academic(model, input_data):
    """
    Uses a trained SVM model to predict degree completion on new academic input.
    Returns a human-readable prediction and confidence score.
    """
    if hasattr(input_data, 'values'):
        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)
    else:
        prediction = model.predict([input_data])
        proba = model.predict_proba([input_data])

    predicted_value = prediction[0]
    confidence = np.max(proba[0])

    if isinstance(predicted_value, str):
        result = "likely to complete a degree" if predicted_value.lower() == "completed" else "unlikely to complete degree"
    else:
        result = "likely to complete a degree" if predicted_value == 1 else "unlikely to complete degree"

    return result, confidence
