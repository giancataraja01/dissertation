from django.shortcuts import render
from django.conf import settings
import os
import pandas as pd
import joblib
import numpy as np

from django.http import HttpResponseForbidden
from .forms import PredictorForm
from upload_dataset.models import StudentRecord, PredictionResult
from .ml.gpt_adviser import generate_advice
from .ml.utils import (
    predict_with_confidence,
    analyze_sentiment_with_confidence,
    analyze_dropout_factors,
    analyze_dropout_factors1,
    predict_svm_academic,  # keep if you still use it elsewhere
)

# ---------- Load base models and artifacts ----------
ML_DIR = os.path.join(settings.BASE_DIR, 'predict', 'ml')

# Base models
dt_model = joblib.load(os.path.join(ML_DIR, 'dt_model.pkl'))
svm_feedback_model = joblib.load(os.path.join(ML_DIR, 'svm_feedback.pkl'))
svm_academic_model = joblib.load(os.path.join(ML_DIR, 'svm_academic.pkl'))
scaler = joblib.load(os.path.join(ML_DIR, 'scaler.pkl'))

# Hybrid configs / models (may not exist on first run)
soft_cfg_path = os.path.join(ML_DIR, 'hybrid_soft_config.pkl')
stack_model_path = os.path.join(ML_DIR, 'hybrid_meta_model.pkl')
stack_cfg_path = os.path.join(ML_DIR, 'hybrid_stack_config.pkl')

soft_cfg = joblib.load(soft_cfg_path) if os.path.exists(soft_cfg_path) else None
stack_meta = joblib.load(stack_model_path) if os.path.exists(stack_model_path) else None
stack_cfg = joblib.load(stack_cfg_path) if os.path.exists(stack_cfg_path) else None

# Metrics (DT / SVM Academic / Hybrids)
dt_metrics = joblib.load(os.path.join(ML_DIR, 'dt_metrics.pkl')) if os.path.exists(os.path.join(ML_DIR, 'dt_metrics.pkl')) else None
svm_acad_metrics = joblib.load(os.path.join(ML_DIR, 'svm_academic_metrics.pkl')) if os.path.exists(os.path.join(ML_DIR, 'svm_academic_metrics.pkl')) else None

soft_metrics = joblib.load(os.path.join(ML_DIR, 'hybrid_soft_metrics.pkl')) if os.path.exists(os.path.join(ML_DIR, 'hybrid_soft_metrics.pkl')) else None
hard_metrics = joblib.load(os.path.join(ML_DIR, 'hybrid_hard_metrics.pkl')) if os.path.exists(os.path.join(ML_DIR, 'hybrid_hard_metrics.pkl')) else None
stack_metrics = joblib.load(os.path.join(ML_DIR, 'hybrid_stack_metrics.pkl')) if os.path.exists(os.path.join(ML_DIR, 'hybrid_stack_metrics.pkl')) else None


def _dt_prob(df_scaled: pd.DataFrame) -> float:
    """Return DT probability for class=1 (Completed) for a single-row df (scaled)."""
    p = dt_model.predict_proba(df_scaled)[:, 1]
    return float(p[0])

def _svm_academic_prob(df_raw: pd.DataFrame) -> float:
    """Return SVM Academic probability for class=1 (Completed) for a single-row df (raw)."""
    p = svm_academic_model.predict_proba(df_raw)[:, 1]
    return float(p[0])


def _hybrid_soft_pred(p_dt: float, p_svm: float):
    """Weighted soft vote using saved weights + threshold. Returns (pred_label, score, info_str)."""
    if soft_cfg is None:
        return None, None, "Soft config not found."
    #w_dt = soft_cfg.get("w_dt", 0.7)
    #w_svm = soft_cfg.get("w_svm", 0.3)
    w_dt=0.8
    w_svm=0.2
    t = soft_cfg.get("threshold", 0.5)
    score = w_dt * p_dt + w_svm * p_svm
    label = int(score >= t)
    msg = f"Soft vote (w_dt={w_dt:.2f}, w_svm={w_svm:.2f}, t={t:.2f})"
    print("Weighted dt",w_dt)
    print("Weighted svm",w_svm)
    print("Babaw Soft socre",score)
    return label, score, msg

def _hybrid_hard_pred(p_dt: float, p_svm: float):
    """Weighted hard vote 3:2 (≈80/20). Returns (pred_label, votes_info)."""
    v_dt = int(p_dt >= 0.5)
    v_svm = int(p_svm >= 0.5)
    votes = 4 * v_dt + 1 * v_svm
    label = int(votes >= 3)  # threshold to favor DT slightly
    msg = f"Hard vote (4*DT + 1*SVM = {votes})"
    return label, msg

def _hybrid_stack_pred(p_dt: float, p_svm: float):
    """Stacking using meta model + threshold. Returns (pred_label, score, info_str)."""
    if stack_meta is None or stack_cfg is None:
        return None, None, "Stacking model/config not found."
    import numpy as np
    val_meta_X = np.array([[p_dt, p_svm]])
    prob = stack_meta.predict_proba(val_meta_X)[:, 1][0]
    t = stack_cfg.get("threshold", 0.5)
    label = int(prob >= t)
    msg = f"Stacking (t={t:.2f})"
    return label, float(prob), msg


def predict(request):

    result = sentiment = risks = risks1 = None

    if request.method == "POST":
        form = PredictorForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data

            # ---------- 1) Sentiment (text SVM) ----------
            sentiment = analyze_sentiment_with_confidence(svm_feedback_model, cd['feedback'])
            sentiment_value = sentiment[1]  # numeric score/prob used as feature

            # ---------- 2) Build raw (unscaled) dataframe for this student ----------
            raw_df = pd.DataFrame([{
                'shs_gpa': float(cd['shs_gpa']),
                'coll_gpa': float(cd['coll_gpa']),
                'age': int(cd['age']),
                'sex': int(cd['sex']),
                'is_boarding': int(cd['is_boarding']),
                'is_living_with_family': int(cd['is_liv_fam']),
                'is_province': int(cd['is_province']),
                'is_scholar': int(cd['is_scholar']),
                'father_educ': int(cd['father_educ']),
                'mother_educ': int(cd['mother_educ']),
                'financial_status': int(cd['financial_status']),
                'sentiments': sentiment_value
            }])

            # ---------- 3) Risk analysis ----------
            stud_id = request.session.get('user_id')
            coll = request.session.get('coll')
            risks1 = analyze_dropout_factors1(raw_df, stud_id)  # uses raw
            risks = analyze_dropout_factors(raw_df)             # if your util expects raw

            # ---------- 4) Prepare model-specific inputs ----------
            # DT needs scaled features
            numerical_cols = [
                'shs_gpa','coll_gpa','age','sex','is_boarding','is_living_with_family',
                'is_province','is_scholar','father_educ','mother_educ','financial_status','sentiments'
            ]
            df_scaled = raw_df.copy()
            df_scaled[numerical_cols] = scaler.transform(df_scaled[numerical_cols])

            # ---------- 5) Single-model predictions ----------
            # If your predict_with_confidence expects scaled (DT), keep as-is:
            result = predict_with_confidence(dt_model, df_scaled)

            # For SVM Academic, ALWAYS pass RAW (pipeline scales internally)
            svm_result = predict_svm_academic(svm_academic_model, raw_df)

            # ---------- 6) Hybrid single-sample predictions ----------
            
            # ---------------- HARD SCORE CALCULATION ----------------
            # result[1] → DT confidence, svm_result[1] → SVM confidence
            p_dt = _dt_prob(df_scaled)
            p_svm = _svm_academic_prob(raw_df)

            v_dt = int(p_dt >= 0.5)
            v_svm = int(p_svm >= 0.5)
        
            hard_votes = 4 * v_dt + 1 * v_svm       # 0..5
            hard_score = hard_votes / 5.0   
            hard_percent = p_dt *.8 + p_svm *.2
            
            print("decision tree: ",p_dt)
            print("svm: ",p_svm)
            print("sentiments: ",sentiment[0])
            print("dt (80%) + svm (20%) = ", p_dt *.8 + p_svm *.2 )
            print("vdt: ",v_dt)
            print("vsm: ",v_svm)
            print("Hard hybrid score: ",hard_score)
            
  #########################################################################
    #     if weighted_score >= 0.6:
    #     code="Likely to complete a degree"
    #     if svm_pred == "Positive":
    #         weighted_pred = (
    #             "You are likely to complete your degree, and your positive attitude supports your success", weighted_score)
    #     elif svm_pred == "Negative":
            
    #         weighted_pred = (
    #             "You are likely to complete your degree, but consider addressing the concerns reflected in your feedback", weighted_score)
    #     else:
            
    #         weighted_pred = (
    #             "You are likely to complete your degree, even though your attitude appears neutral", weighted_score)
    # elif weighted_score >= 0.4:
    #     code="Moderately likely to complete a degree"
    #     if svm_pred == "Positive":            
    #         weighted_pred = (
    #             "You are moderately likely to complete your degree, and your positive outlook is helpful", weighted_score)
    #     elif svm_pred == "Negative":            
    #         weighted_pred = (
    #             "You are moderately likely to complete your degree, but your negative sentiment may be a risk factor", weighted_score)
    #     else:            
    #         weighted_pred = (
    #             "You are moderately likely to complete your degree, though your neutral feedback suggests uncertainty", weighted_score)
    # else:
    #     code="Unlikely to complete degree"
    #     if svm_pred == "Positive":
            
    #         weighted_pred = (
    #             "You are currently at risk of not completing your degree, but your positive attitude may help you succeed ", weighted_score)
    #     elif svm_pred == "Negative":
            
    #         weighted_pred = (
    #             "There is a risk of not completing your degree. Consider seeking support to address academic and emotional challenges", weighted_score)
    #     else:            
    #         weighted_pred = (
    #             "You may be at risk of not completing your degree. Try to engage more actively in your studies and seek guidance", weighted_score)

    # if average_score >= 0.6:
    #     average_pred = ("Likely to complete a degree", average_score)
    # elif average_score >= 0.4:
    #     average_pred = (
    #         "Moderately likely to complete a degree", average_score)
    # else:
    #     average_pred = ("Unlikely to complete degree", average_score)

    # return weighted_pred, weighted_score, average_pred, average_score, code
  #########################################################################

            # # Assign labels based on thresholds hard vote
            # if hard_score >= 0.75:
            #     hard_result = "Likely to complete a degree"
            #     #code = "Likely to complete a degree"
            #     if sentiment[0] == "Positive":
            #         hard_pred = (
            #             "You are likely to complete your degree, and your positive attitude supports your success")
            #     elif sentiment[0] == "Negative":
            #         hard_pred = (
            #             "You are likely to complete your degree, but consider addressing the concerns reflected in your feedback")
            #     else:                    
            #         hard_pred = (
            #             "You are likely to complete your degree, even though your attitude appears neutral")
            
            # elif hard_score >= 0.50:
            #     hard_result = "Moderately likely to complete a degree"
            #     if sentiment == "Positive":            
            #         hard_pred = (
            #             "You are moderately likely to complete your degree, and your positive outlook is helpful")
            #     elif sentiment == "Negative":            
            #         hard_pred = (
            #             "You are moderately likely to complete your degree, but your negative sentiment may be a risk factor")
            #     else:            
            #         hard_pred = (
            #             "You are moderately likely to complete your degree, though your neutral feedback suggests uncertainty")

            # else:
            #     hard_result = "Unlikely to complete a degree"

            #     if sentiment == "Positive":
                             
            #         hard_pred = (
            #             "You are currently at risk of not completing your degree, but your positive attitude may help you succeed ")
            #     elif sentiment == "Negative":
                        
            #             hard_pred = (
            #                 "There is a risk of not completing your degree. Consider seeking support to address academic and emotional challenges")
            #     else:            
            #             hard_pred = (
            #                 "You may be at risk of not completing your degree. Try to engage more actively in your studies and seek guidance")
            # Assign labels based on thresholds hard vote

            # if hard_score >= 0.75:
            #     hard_result = "Likely to complete a degree"
            #     #code = "Likely to complete a degree"
            #     if sentiment[0] == "Positive":
            #         hard_pred = (
            #             "You are likely to complete your degree, and your positive attitude supports your success")
            #     elif sentiment[0] == "Negative":
            #         hard_pred = (
            #             "You are likely to complete your degree, but consider addressing the concerns reflected in your feedback")
            #     else:                    
            #         hard_pred = (
            #             "You are likely to complete your degree, even though your attitude appears neutral")
            
            # elif hard_score >= 0.50:
            #     hard_result = "Moderately likely to complete a degree"
            #     if sentiment == "Positive":            
            #         hard_pred = (
            #             "You are moderately likely to complete your degree, and your positive outlook is helpful")
            #     elif sentiment == "Negative":            
            #         hard_pred = (
            #             "You are moderately likely to complete your degree, but your negative sentiment may be a risk factor")
            #     else:            
            #         hard_pred = (
            #             "You are moderately likely to complete your degree, though your neutral feedback suggests uncertainty")

            # else:
            #     hard_result = "Unlikely to complete a degree"

            #     if sentiment == "Positive":
                             
            #         hard_pred = (
            #             "You are currently at risk of not completing your degree, but your positive attitude may help you succeed ")
            #     elif sentiment == "Negative":
                        
            #             hard_pred = (
            #                 "There is a risk of not completing your degree. Consider seeking support to address academic and emotional challenges")
            #     else:            
            #             hard_pred = (
            #                 "You may be at risk of not completing your degree. Try to engage more actively in your studies and seek guidance")
           
            soft_label, soft_score, soft_info = _hybrid_soft_pred(p_dt, p_svm)
            hard_label, hard_info = _hybrid_hard_pred(p_dt, p_svm)
            stack_label, stack_score, stack_info = _hybrid_stack_pred(p_dt, p_svm)
            # print("Hard hybrid result: ",hard_result)
            # Optional: human-friendly messages
            def _label_to_msg(lbl):
                return "Likely to complete a degree" if lbl == 1 else "Unlikely to complete a degree"

            # ----------------- SOFT VOTING
            
            # if soft_cfg is None:
            #     return None, None, "Soft config not found."
            # w_dt = soft_cfg.get("w_dt", 0.7)
            # w_svm = soft_cfg.get("w_svm", 0.3)
            # t = soft_cfg.get("threshold", 0.5)
            # score = w_dt * p_dt + w_svm * p_svm
            # label = int(score >= t)
            # msg = f"Soft vote (w_dt={w_dt:.2f}, w_svm={w_svm:.2f}, t={t:.2f})"
            # print("Soft socre",score)
                
            soft_msg = _label_to_msg(soft_label) if soft_label is not None else "N/A"
            hard_msg = _label_to_msg(hard_label) if hard_label is not None else "N/A"
            stack_msg = _label_to_msg(stack_label) if stack_label is not None else "N/A"
            # print("Soft Score is: ",score)
            print("The Soft Score is: ",soft_score)            
            sentiment = sentiment[0]
            print("Sentiment is: ",sentiment)
            if soft_score >= 0.75:
                soft_result = "Likely to complete a degree"
                #code = "Likely to complete a degree"
                if sentiment[0] == "Positive":
                    soft_pred = (
                        "You are likely to complete your degree, and your positive attitude supports your success")
                elif sentiment[0] == "Negative":
                    soft_pred = (
                        "You are likely to complete your degree, but consider addressing the concerns reflected in your feedback")
                else:                    
                    soft_pred = (
                        "You are likely to complete your degree, even though your attitude appears neutral")
            
            elif soft_score >= 0.50:
                soft_result = "Moderately likely to complete a degree"
                if sentiment == "Positive":            
                    soft_pred = (
                        "You are moderately likely to complete your degree, and your positive outlook is helpful")
                elif sentiment == "Negative":            
                    soft_pred = (
                        "You are moderately likely to complete your degree, but your negative sentiment may be a risk factor")
                else:            
                    soft_pred = (
                        "You are moderately likely to complete your degree, though your neutral feedback suggests uncertainty")

            else:
                soft_result = "Unlikely to complete a degree"

                if sentiment == "Positive":
                             
                    soft_pred = (
                        "You are currently at risk of not completing your degree, but your positive attitude may help you succeed ")
                elif sentiment == "Negative":
                        
                        soft_pred = (
                            "There is a risk of not completing your degree. Consider seeking support to address academic and emotional challenges")
                else:            
                        soft_pred = (
                            "You may be at risk of not completing your degree. Try to engage more actively in your studies and seek guidance")

            print("Label Result: ", soft_result)
            print("Label Result: ", soft_pred)
            # ---------- 7) Save to DB (keep your structure; adapt as needed) ----------
            
            try:
                PredictionResult.objects.create(
                    stud_id = stud_id,
                    coll = coll,
                    course = request.session.get('course'),
                    svm_prediction = sentiment,
                    hybrid_result = soft_msg if soft_label is not None else None,
                    hybrid_score = soft_score if soft_score is not None else 0.0,   # never None
                    average_result = hard_msg if hard_label is not None else None,  # your “hard vote” text
                    average_score = hard_score,                                      # <-- key change
                    code = soft_result if soft_label is not None else 'N/A',
                    # code = hard_result if soft_label is not None else 'N/A',
                    # //code = 'SOFT' if soft_label is not None else 'N/A',
                    feedback = cd['feedback']
)
            except StudentRecord.DoesNotExist:
                print(f"❌ No StudentRecord found with ID {stud_id}")

            # ---------- 8) Advice ----------
            advice = generate_advice(cd['coll_gpa'], sentiment[0], risks1)

            # ---------- 9) Render ----------
            # NOTE: Keeping your old average/weighted fields but **also** showing real hybrid metrics.
            
            context = {
                "result": result,
                "sentiment": sentiment,
                "svm_academic": svm_result[0],
                "svm_academic_conf": svm_result[1] * 100,

                "hybrid_soft_label": soft_msg,
                "hybrid_soft_score": None if soft_score is None else soft_score * 100,
                "hybrid_soft_info": soft_info,

                # "hybrid_hard_label": hard_msg,
                "hybrid_hard_label": soft_pred,
                "hybrid_hard_info": soft_info,
                "hybrid_hard_score": soft_score * 100,
                "hybrid_hard_percent" :soft_score * 100,
                #  -------------- HARD VOTING -----------------#
                # "hybrid_hard_label": hard_msg,
                # "hybrid_hard_label": hard_pred,
                # "hybrid_hard_info": hard_info,
                # "hybrid_hard_score": hard_score * 100,
                # "hybrid_hard_percent" :hard_percent * 100,

                "hybrid_stack_label": stack_msg,
                "hybrid_stack_score": None if stack_score is None else stack_score * 100,
                "hybrid_stack_info": stack_info,

                # Base model metrics
                "dt_acc": dt_metrics['accuracy'] * 100 if dt_metrics else None,
                "dt_prec": dt_metrics['weighted avg']['precision'] * 100 if dt_metrics else None,
                "dt_rec": dt_metrics['weighted avg']['recall'] * 100 if dt_metrics else None,
                "dt_f1": dt_metrics['weighted avg']['f1-score'] * 100 if dt_metrics else None,

                "svm_acc": svm_acad_metrics['accuracy'] * 100 if svm_acad_metrics else None,
                "svm_prec": svm_acad_metrics['weighted avg']['precision'] * 100 if svm_acad_metrics else None,
                "svm_rec": svm_acad_metrics['weighted avg']['recall'] * 100 if svm_acad_metrics else None,
                "svm_f1": svm_acad_metrics['weighted avg']['f1-score'] * 100 if svm_acad_metrics else None,

                # TRUE hybrid metrics (from training)
                "hybrid_soft_acc": soft_metrics['accuracy'] if soft_metrics else None,
                "hybrid_soft_prec": soft_metrics['precision'] if soft_metrics else None,
                "hybrid_soft_rec": soft_metrics['recall'] if soft_metrics else None,
                "hybrid_soft_f1": soft_metrics['f1'] if soft_metrics else None,

                "hybrid_hard_acc": hard_metrics['accuracy'] if hard_metrics else None,
                "hybrid_hard_prec": hard_metrics['precision'] if hard_metrics else None,
                "hybrid_hard_rec": hard_metrics['recall'] if hard_metrics else None,
                "hybrid_hard_f1": hard_metrics['f1'] if hard_metrics else None,

                "hybrid_stack_acc": stack_metrics['accuracy'] if stack_metrics else None,
                "hybrid_stack_prec": stack_metrics['precision'] if stack_metrics else None,
                "hybrid_stack_rec": stack_metrics['recall'] if stack_metrics else None,
                "hybrid_stack_f1": stack_metrics['f1'] if stack_metrics else None,

                # Keep your legacy “average/weighted” placeholders (but consider removing in the template)
                "hybrid_acc_ave": None,
                "hybrid_acc_wei": None,
                "hybrid_prec_ave": None,
                "hybrid_prec_wei": None,
                "hybrid_rec_ave": None,
                "hybrid_rec_wei": None,
                "hybrid_f1_ave": None,
                "hybrid_f1_wei": None,

                "risks": risks,
                "risks1": risks1,
                "advice": advice,
            }

            return render(request, "predict/display_predict.html", context)

    else:
        form = PredictorForm()

    return render(request, "predict/display.html", {"form": form})


##########################################################################

# from django.shortcuts import render
# from django.conf import settings
# import os
# import pandas as pd
# import joblib
# from .forms import PredictorForm
# from upload_dataset.models import StudentRecord, PredictionResult
# from .ml.gpt_adviser import generate_advice
# from .ml.utils import (
#     predict_with_confidence,
#     analyze_sentiment_with_confidence,
#     calculate_combined_predictions,
#     analyze_dropout_factors,
#     analyze_dropout_factors1,   
#     predict_svm_academic,
#     )

# # Load models and scaler
# dt_model = joblib.load(os.path.join(
#     settings.BASE_DIR, 'predict/ml/dt_model.pkl'))
# svm_model = joblib.load(os.path.join(
#     settings.BASE_DIR, 'predict/ml/svm_feedback.pkl'))
# scaler = joblib.load(os.path.join(settings.BASE_DIR, 'predict/ml/scaler.pkl'))

# svm_academic_model = joblib.load(os.path.join(
#     settings.BASE_DIR, 'predict/ml/svm_academic.pkl'))
# # Load evaluation metrics

# dt_accuracy = svm_accuracy = svm_metrics= None
# def predict(request):
    
#     result = sentiment = hybrid = risks = risks1 = None

#     if request.method == "POST":
#         form = PredictorForm(request.POST)
#         if form.is_valid():
#             cd = form.cleaned_data
#             sentiment = analyze_sentiment_with_confidence(svm_model, cd['feedback'])
#             sentiment_value = sentiment[1]
#             print('prediction sentiment',sentiment)
#             print('prediction sentiment',sentiment_value)
#             # Step 1: Create lowercase DataFrame from form
#             raw_df = pd.DataFrame([{
#                 'shs_gpa': float(cd['shs_gpa']),
#                 'coll_gpa': float(cd['coll_gpa']),
#                 'age': int(cd['age']),
#                 'sex': int(cd['sex']),
#                 'is_boarding': int(cd['is_boarding']),
#                 'is_living_with_family': int(cd['is_liv_fam']),
#                 'is_province': int(cd['is_province']),
#                 'is_scholar': int(cd['is_scholar']),
#                 'father_educ': int(cd['father_educ']),
#                 'mother_educ': int(cd['mother_educ']),
#                 'financial_status': int(cd['financial_status']),
#                 'sentiments': sentiment_value
#             }])
                       
#             # Step 2: Analyze risks using raw (unscaled) values
#             stud_id = request.session['user_id']
#             coll = request.session['coll']
#             risks1 = analyze_dropout_factors1(raw_df, stud_id)

#             # Step 4: Create model input df (exclude 'sentiments' if model wasn't trained with it)
#             # df = raw_df.drop(columns=['sentiments'])
#             # Step 3: Scale numerical features
#             #numerical_cols=list(df.columns)
#             #print("The columns:",numerical_cols)
#             df = raw_df.copy()
#             # numerical_cols = ['shs_gpa',
#             #                   'coll_gpa', 'age', 'sex', 'is_boarding', 'is_living_with_family',
#             #                   'is_province', 'is_scholar', 'father_educ',
#             #                   'mother_educ', 'financial_status']
#             numerical_cols = ['shs_gpa',
#                                'coll_gpa', 'age', 'sex', 'is_boarding', 'is_living_with_family',
#                                'is_province', 'is_scholar', 'father_educ',
#                                'mother_educ', 'financial_status','sentiments']
#             df[numerical_cols] = scaler.transform(df[numerical_cols])

           
#             # Step 4: Run predictions

#             # print(dt_model, df) weighted_pred, average_pred, weighted_score, average_score
#             result = predict_with_confidence(dt_model, df)           
#             # Predict degree completion using academic SVM
#             svm_result = predict_svm_academic(svm_academic_model, df)

#             hybrid = calculate_combined_predictions(
#                 result[0], result[1], sentiment[0], sentiment[1]) 
#             risks = analyze_dropout_factors(df) 
#             senti = cd['feedback']   
#             advice = generate_advice(cd['coll_gpa'], sentiment[0], risks1)
#             # advice = generate_advice(cd['coll_gpa'], sentiment[0], cd['feedback'])

#             # saving the prediction result
#             try:
#                  weighted_pred, weighted_score, average_pred, average_score, code = hybrid
#                  PredictionResult.objects.create(
#                     stud_id = request.session['user_id'],
#                     coll = request.session['coll'],
#                     course = request.session['course'],
#                     # dt_prediction=result[0],           # e.g., "likely to complete a degree"
#                     # dt_confidence=result[1],           # float value
#                     svm_prediction=sentiment[0],       # e.g., "Positive"
#                     # svm_confidence=sentiment[1],       # float value
#                      hybrid_result=hybrid[0][0],        # descriptive string
#                      hybrid_score=hybrid[1],            # weighted score
#                     average_result=hybrid[2][0],       # "likely to complete a degree"
#                     average_score=hybrid[2][1],        # average score
#                     code = code,
#                     feedback = senti
#                 )
#             except StudentRecord.DoesNotExist:
#                 print(f"❌ No StudentRecord found with ID {stud_id}")

#             # end of prediction result                
            
#             dt_metrics_path = os.path.join(settings.BASE_DIR, 'predict/ml/dt_metrics.pkl')
#             svm_metrics_path = os.path.join(settings.BASE_DIR, 'predict/ml/svm_metrics.pkl')
#             svm_metrics_path2 = os.path.join(settings.BASE_DIR, 'predict/ml/svm_academic_metrics.pkl')
#             if os.path.exists(dt_metrics_path):
#                 dt_metrics = joblib.load(dt_metrics_path)
#             if os.path.exists(svm_metrics_path):
#                 svm_metrics = joblib.load(svm_metrics_path)
#             if os.path.exists(svm_metrics_path):
#                 svm_metrics = joblib.load(svm_metrics_path2)

#             print("prediction result", result)
#             print("\nprediction sentiment", sentiment)
#             print("\nprediction hybrid weighted", hybrid[0])
#             print("\nprediction hybrid average", hybrid[2])        


#             return render(request, "predict/display_predict.html", {
#                 "result": result,
#                 "sentiment": sentiment,
#                 "hybrid": hybrid[0],
#                 "weighted": hybrid[1]*100,
#                 "ave_mess": hybrid[2],
#                 "ave":hybrid[3]*100,
#                 "risks": risks,
#                 "risks1": risks1,
#                 "dt_acc": dt_metrics['accuracy'] * 100,
#                 "dt_prec": dt_metrics['weighted avg']['precision'] * 100,
#                 "dt_rec": dt_metrics['weighted avg']['recall'] * 100,
#                 "dt_f1": dt_metrics['weighted avg']['f1-score'] * 100,

#                 "svm_acc": svm_metrics['accuracy'] * 100,
#                 "svm_prec": svm_metrics['weighted avg']['precision'] * 100,
#                 "svm_rec": svm_metrics['weighted avg']['recall'] * 100,
#                 "svm_f1": svm_metrics['weighted avg']['f1-score'] * 100,

#                 "hybrid_acc_ave": (dt_metrics['accuracy'] + svm_metrics['accuracy']) / 2 * 100,
#                 "hybrid_acc_wei": (dt_metrics['accuracy'] *.8 + svm_metrics['accuracy']*.2) * 100,
                
#                 "hybrid_prec_ave":(dt_metrics['weighted avg']['precision'] + svm_metrics['weighted avg']['precision']) / 2 * 100,
#                 "hybrid_prec_wei":(dt_metrics['weighted avg']['precision'] *.8 + svm_metrics['weighted avg']['precision'] * .2)* 100,
                
#                 "hybrid_rec_ave": (dt_metrics['weighted avg']['recall'] + svm_metrics['weighted avg']['recall']) /2 * 100,
#                 "hybrid_rec_wei": (dt_metrics['weighted avg']['recall']*.8 + svm_metrics['weighted avg']['recall'] *.2) * 100,

#                 "hybrid_f1_ave": (dt_metrics['weighted avg']['f1-score'] + svm_metrics['weighted avg']['f1-score']) / 2 * 100,
#                 "hybrid_f1_wei": (dt_metrics['weighted avg']['f1-score']*.8 + svm_metrics['weighted avg']['f1-score'] *.2) * 100,
#                 "advice": advice,
#                 "svm_academic": svm_result[0],
#                 "svm_academic_conf": svm_result[1] * 100,

#             })
#     else:
#         form = PredictorForm()

#     return render(request, "predict/display.html", {
#         "form": form,
#     })
