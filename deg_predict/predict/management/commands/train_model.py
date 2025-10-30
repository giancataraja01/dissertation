from django.core.management.base import BaseCommand
from upload_dataset.models import StudentRecord
from django.conf import settings

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score, f1_score
)

import pandas as pd
import numpy as np
import joblib
import os


class Command(BaseCommand):
    help = "Train Decision Tree, SVM (academic & sentiment), and Hybrid models (soft, hard, stacking) using data from the database"

    def handle(self, *args, **kwargs):
        # ---------- Paths / cleanup ----------
        model_dir = os.path.join(settings.BASE_DIR, 'predict', 'ml')
        os.makedirs(model_dir, exist_ok=True)

        model_files = [
            # base models + metrics
            'dt_model.pkl', 'dt_metrics.pkl', 'scaler.pkl',
            'svm_feedback.pkl', 'svm_metrics.pkl',
            'svm_academic.pkl', 'svm_academic_metrics.pkl',
            # hybrid artifacts
            'hybrid_soft_config.pkl', 'hybrid_soft_metrics.pkl',
            'hybrid_hard_metrics.pkl',
            'hybrid_meta_model.pkl', 'hybrid_stack_config.pkl', 'hybrid_stack_metrics.pkl',
            # label mapping
            'label_mapping.pkl'
        ]

        for file in model_files:
            path = os.path.join(model_dir, file)
            if os.path.exists(path):
                os.remove(path)
                self.stdout.write(self.style.WARNING(f"🗑️ Deleted: {file}"))

        # ---------- Load data ----------
        self.stdout.write("📦 Loading data from database...")
        records = StudentRecord.objects.all().values()
        df = pd.DataFrame(list(records))

        if df.empty:
            self.stdout.write(self.style.ERROR("❌ No data found in database."))
            return

        self.stdout.write(self.style.SUCCESS(f"✅ Loaded {len(df)} records."))

        # ---------- Normalize column names ----------
        df = df.rename(columns={
            'SHS_GPA_AVERAGE': 'shs_gpa',
            'COLL_GPA_AVERAGE': 'coll_gpa',
            'AGE': 'age',
            'SEX': 'sex',
            'IS_BOARDING': 'is_boarding',
            'IS_LIVING_WITH_FAMILY': 'is_living_with_family',
            'IS_PROVINCE': 'is_province',
            'IS_SCHOLAR': 'is_scholar',
            'FATHER_EDUC_BACKGROUND': 'father_educ',
            'MOTHER_EDUC_BACKGROUND': 'mother_educ',
            'PARENTS_MONTHLY_INCOME': 'financial_status',
            'IS_GRADUATE': 'is_graduate',
            'FEEDBACK': 'feedback',
            'SENTIMENTS': 'sentiments',
            'DEPARTMENT': 'department'
        })

        if 'is_graduate' not in df.columns:
            self.stdout.write(self.style.ERROR("❌ 'is_graduate' column missing."))
            return

        # ---------- Normalize target labels to 0/1 ----------
        # Accept common string variants; keep numeric as-is
        label_map_display = {'Completed': 1, 'Not Completed': 0}
        inv_label_map_display = {v: k for k, v in label_map_display.items()}

        if df['is_graduate'].dtype == 'O':
            s = df['is_graduate'].astype(str).str.strip().str.lower()
            mapped = s.map({
                'completed': 1,
                'not completed': 0,
                'not_completed': 0,
                'notcompleted': 0,
                '1': 1,
                '0': 0,
                'true': 1,
                'false': 0
            })
            df['is_graduate'] = mapped
        else:
            # Coerce numeric to int 0/1
            df['is_graduate'] = df['is_graduate'].astype(float).round().astype(int)

        if df['is_graduate'].isnull().any():
            unknowns = df.loc[df['is_graduate'].isnull(), 'is_graduate'].shape[0]
            raise ValueError(f"Found unknown labels in 'is_graduate' ({unknowns} rows). "
                             f"Please standardize to Completed / Not Completed or 1/0.")

        # Persist mapping for UI (optional)
        joblib.dump(
            {'label_map': {'Completed': 1, 'Not Completed': 0},
             'inv_label_map': {1: 'Completed', 0: 'Not Completed'}},
            os.path.join(model_dir, 'label_mapping.pkl')
        )

        # ---------- Unified split (so predictions align across models) ----------
        X_base = df.drop(columns=['is_graduate', 'id', 'department', 'feedback'], errors='ignore').copy()
        if 'sentiments' not in X_base.columns and 'sentiments' in df.columns:
            X_base['sentiments'] = df['sentiments']

        y_base = df['is_graduate']

        # Only stratify if both classes present
        stratify_arg = y_base if y_base.nunique() > 1 else None

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_base, y_base, test_size=0.2, random_state=42, stratify=stratify_arg
        )

        # ---------- Decision Tree (uses saved StandardScaler) ----------
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)

            dt_model = DecisionTreeClassifier(
                max_depth=10,
                min_samples_leaf=2,
                min_samples_split=4,
                min_impurity_decrease=0.01,
                ccp_alpha=0.1,
                random_state=42
            )
            dt_model.fit(X_train_scaled, y_train)
            y_pred_dt = dt_model.predict(X_test_scaled)

            dt_metrics = classification_report(y_test, y_pred_dt, output_dict=True)
            acc = dt_metrics.get("accuracy", 0) * 100
            precision = dt_metrics['weighted avg']['precision'] * 100
            recall = dt_metrics['weighted avg']['recall'] * 100
            f1 = dt_metrics['weighted avg']['f1-score'] * 100

            self.stdout.write(self.style.SUCCESS(f"✅ Decision Tree Accuracy: {acc:.2f}%"))
            self.stdout.write(self.style.SUCCESS(f"✅ Precision: {precision:.2f}%"))
            self.stdout.write(self.style.SUCCESS(f"✅ Recall: {recall:.2f}%"))
            self.stdout.write(self.style.SUCCESS(f"✅ F1-score: {f1:.2f}%"))

            joblib.dump(dt_model, os.path.join(model_dir, 'dt_model.pkl'))
            joblib.dump(dt_metrics, os.path.join(model_dir, 'dt_metrics.pkl'))
            joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
            self.stdout.write(self.style.SUCCESS("✅ Decision Tree model, metrics, and scaler saved."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ Error during Decision Tree training: {e}"))
            return  # hybrids depend on DT

        # ---------- SVM Academic (pipeline handles its own scaling) ----------
        try:
            svm_academic = make_pipeline(
                StandardScaler(),
                SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=42)
            )

            svm_academic.fit(X_train_raw, y_train)
            y_pred_svm = svm_academic.predict(X_test_raw)

            svm_academic_metrics = classification_report(y_test, y_pred_svm, output_dict=True)

            acc = svm_academic_metrics.get("accuracy", 0) * 100
            prec = svm_academic_metrics["weighted avg"]["precision"] * 100
            rec = svm_academic_metrics["weighted avg"]["recall"] * 100
            f1 = svm_academic_metrics["weighted avg"]["f1-score"] * 100

            self.stdout.write(self.style.SUCCESS(f"📊 SVM Academic Accuracy: {acc:.2f}%"))
            self.stdout.write(self.style.SUCCESS(f"📊 SVM Academic Precision: {prec:.2f}%"))
            self.stdout.write(self.style.SUCCESS(f"📊 SVM Academic Recall: {rec:.2f}%"))
            self.stdout.write(self.style.SUCCESS(f"📊 SVM Academic F1-score: {f1:.2f}%"))

            joblib.dump(svm_academic, os.path.join(model_dir, 'svm_academic.pkl'))
            joblib.dump(svm_academic_metrics, os.path.join(model_dir, 'svm_academic_metrics.pkl'))
            self.stdout.write(self.style.SUCCESS("✅ SVM Academic model and metrics saved."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ SVM Academic model training failed: {e}"))
            return  # hybrids depend on SVM too

        # ---------- SVM Sentiment (feedback -> sentiments) ----------
        if 'feedback' in df.columns and 'sentiments' in df.columns:
            try:
                X_text = df['feedback'].fillna("").astype(str)
                y_sent = df['sentiments']
                X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
                    X_text, y_sent, test_size=0.2, random_state=42
                )

                sentiment_pipeline = make_pipeline(
                    TfidfVectorizer(),
                    SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=42)
                )

                sentiment_pipeline.fit(X_train_t, y_train_t)
                y_pred_t = sentiment_pipeline.predict(X_test_t)

                sentiment_metrics = classification_report(y_test_t, y_pred_t, output_dict=True)
                if sentiment_metrics:
                    acc = sentiment_metrics.get("accuracy", 0) * 100
                    prec = sentiment_metrics["weighted avg"]["precision"] * 100
                    rec = sentiment_metrics["weighted avg"]["recall"] * 100
                    f1 = sentiment_metrics["weighted avg"]["f1-score"] * 100

                    self.stdout.write(self.style.SUCCESS(f"🗣️ SVM Sentiment Accuracy: {acc:.2f}%"))
                    self.stdout.write(self.style.SUCCESS(f"🗣️ SVM Sentiment Precision: {prec:.2f}%"))
                    self.stdout.write(self.style.SUCCESS(f"🗣️ SVM Sentiment Recall: {rec:.2f}%"))
                    self.stdout.write(self.style.SUCCESS(f"🗣️ SVM Sentiment F1-score: {f1:.2f}%"))

                joblib.dump(sentiment_pipeline, os.path.join(model_dir, 'svm_feedback.pkl'))
                joblib.dump(sentiment_metrics, os.path.join(model_dir, 'svm_metrics.pkl'))
                self.stdout.write(self.style.SUCCESS("✅ SVM Sentiment model and metrics saved."))

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"❌ SVM Sentiment model training failed: {e}"))

        # ---------- Hybrids (Soft vote, Hard vote, Stacking) ----------
        try:
            # aligned probabilities/preds on the same test split
            dt_prob = dt_model.predict_proba(X_test_scaled)[:, 1]          # DT uses scaled
            svm_prob = svm_academic.predict_proba(X_test_raw)[:, 1]        # SVM pipeline uses raw

            dt_pred = (dt_prob >= 0.5).astype(int)
            svm_pred = (svm_prob >= 0.5).astype(int)

            def eval_and_log(name, y_true, y_pred, out_pkl):
                metrics = {
                    "accuracy": 100 * accuracy_score(y_true, y_pred),
                    "precision": 100 * precision_score(y_true, y_pred, zero_division=0),
                    "recall": 100 * recall_score(y_true, y_pred, zero_division=0),
                    "f1": 100 * f1_score(y_true, y_pred, zero_division=0),
                }
                self.stdout.write(self.style.SUCCESS(
                    f"🤝 {name} -> Acc {metrics['accuracy']:.2f}% | "
                    f"P {metrics['precision']:.2f}% | R {metrics['recall']:.2f}% | F1 {metrics['f1']:.2f}%"
                ))
                joblib.dump(metrics, os.path.join(model_dir, out_pkl))
                return metrics

            # ---- (A) Soft voting with threshold tuning ----
            W_DT, W_SVM = 0.8, 0.2  # tune later if needed
            hybrid_prob = W_DT * dt_prob + W_SVM * svm_prob

            thresholds = np.linspace(0.2, 0.8, 61)
            best_t, best_f1 = 0.5, -1
            for t in thresholds:
                pred = (hybrid_prob >= t).astype(int)
                f1 = f1_score(y_test, pred, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, t

            hybrid_soft_pred = (hybrid_prob >= best_t).astype(int)
            _ = eval_and_log(
                f"Hybrid SOFT (w={W_DT:.1f}/{W_SVM:.1f}, t={best_t:.2f})",
                y_test, hybrid_soft_pred, 'hybrid_soft_metrics.pkl'
            )
            joblib.dump({"w_dt": W_DT, "w_svm": W_SVM, "threshold": float(best_t)},
                        os.path.join(model_dir, 'hybrid_soft_config.pkl'))

            # ---- (B) Hard voting (weighted 3:2 ~ 60/40) ----
            hybrid_hard_pred = ((4 * dt_pred + 1 * svm_pred) >= 3).astype(int)
            _ = eval_and_log("Hybrid HARD (4:1 votes)", y_test, hybrid_hard_pred, 'hybrid_hard_metrics.pkl')

            # ---- (C) Stacking with OOF predictions ----
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            oof_meta_X = np.zeros((len(X_train_raw), 2))
            oof_y = y_train.values if hasattr(y_train, "values") else y_train

            for tr_idx, va_idx in kf.split(X_train_raw, y_train):
                X_tr_raw, X_va_raw = X_train_raw.iloc[tr_idx], X_train_raw.iloc[va_idx]
                y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

                # DT fold
                scaler_k = StandardScaler()
                X_tr_scaled = scaler_k.fit_transform(X_tr_raw)
                X_va_scaled = scaler_k.transform(X_va_raw)

                dt_k = DecisionTreeClassifier(
                    max_depth=10, min_samples_leaf=2, min_samples_split=4,
                    min_impurity_decrease=0.01, ccp_alpha=0.1, random_state=42
                )
                dt_k.fit(X_tr_scaled, y_tr)
                p_dt = dt_k.predict_proba(X_va_scaled)[:, 1]

                # SVM fold
                svm_k = make_pipeline(
                    StandardScaler(),
                    SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=42)
                )
                svm_k.fit(X_tr_raw, y_tr)
                p_svm = svm_k.predict_proba(X_va_raw)[:, 1]

                oof_meta_X[va_idx, 0] = p_dt
                oof_meta_X[va_idx, 1] = p_svm

            meta = LogisticRegression(max_iter=1000, solver='lbfgs')
            meta.fit(oof_meta_X, oof_y)

            # meta validate on held-out test split
            val_meta_X = np.column_stack((
                dt_model.predict_proba(X_test_scaled)[:, 1],
                svm_academic.predict_proba(X_test_raw)[:, 1]
            ))
            meta_prob = meta.predict_proba(val_meta_X)[:, 1]

            # threshold tuning for stacking
            best_t_meta, best_f1_meta = 0.5, -1
            for t in thresholds:
                pred = (meta_prob >= t).astype(int)
                f1 = f1_score(y_test, pred, zero_division=0)
                if f1 > best_f1_meta:
                    best_f1_meta, best_t_meta = f1, t

            meta_pred = (meta_prob >= best_t_meta).astype(int)
            _ = eval_and_log(
                f"Hybrid STACK (t={best_t_meta:.2f})",
                y_test, meta_pred, 'hybrid_stack_metrics.pkl'
            )
            joblib.dump(meta, os.path.join(model_dir, 'hybrid_meta_model.pkl'))
            joblib.dump({"threshold": float(best_t_meta)}, os.path.join(model_dir, 'hybrid_stack_config.pkl'))

            self.stdout.write(self.style.SUCCESS("✅ Hybrid models (soft, hard, stacking) trained and saved."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ Hybrid training failed: {e}"))

        self.stdout.write(self.style.SUCCESS("🏁 Model training complete."))


#############################################################
# from django.core.management.base import BaseCommand
# from upload_dataset.models import StudentRecord
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.pipeline import make_pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from django.conf import settings
# from sklearn.metrics import classification_report
# import pandas as pd
# import joblib
# import os


# class Command(BaseCommand):
#     help = "Train Decision Tree and SVM models using data from the database"

#     def handle(self, *args, **kwargs):
#         model_dir = os.path.join(settings.BASE_DIR, 'predict', 'ml')
#         os.makedirs(model_dir, exist_ok=True)

#         model_files = [
#             'dt_model.pkl', 'dt_metrics.pkl', 'scaler.pkl',
#             'svm_feedback.pkl', 'svm_metrics.pkl',
#             'svm_academic.pkl', 'svm_academic_metrics.pkl'
#         ]

#         for file in model_files:
#             path = os.path.join(model_dir, file)
#             if os.path.exists(path):
#                 os.remove(path)
#                 self.stdout.write(self.style.WARNING(f"🗑️ Deleted: {file}"))

#         self.stdout.write("📦 Loading data from database...")
#         records = StudentRecord.objects.all().values()
#         df = pd.DataFrame(list(records))

#         if df.empty:
#             self.stdout.write(self.style.ERROR("❌ No data found in database."))
#             return

#         self.stdout.write(self.style.SUCCESS(f"✅ Loaded {len(df)} records."))

#         df = df.rename(columns={
#             'SHS_GPA_AVERAGE': 'shs_gpa',
#             'COLL_GPA_AVERAGE': 'coll_gpa',
#             'AGE': 'age',
#             'SEX': 'sex',
#             'IS_BOARDING': 'is_boarding',
#             'IS_LIVING_WITH_FAMILY': 'is_living_with_family',
#             'IS_PROVINCE': 'is_province',
#             'IS_SCHOLAR': 'is_scholar',
#             'FATHER_EDUC_BACKGROUND': 'father_educ',
#             'MOTHER_EDUC_BACKGROUND': 'mother_educ',
#             'PARENTS_MONTHLY_INCOME': 'financial_status',
#             'IS_GRADUATE': 'is_graduate',
#             'FEEDBACK': 'feedback',
#             'SENTIMENTS': 'sentiments',
#             'DEPARTMENT': 'department'
#         })

#         # --- Decision Tree Training ---
#         if 'is_graduate' in df.columns:
#             try:
#                 X = df.drop(columns=['is_graduate', 'id', 'department', 'feedback'], errors='ignore')
#                 if 'sentiments' not in X.columns and 'sentiments' in df.columns:
#                     X['sentiments'] = df['sentiments']

#                 y = df['is_graduate']
#                 scaler = StandardScaler()
#                 X_scaled = scaler.fit_transform(X)

#                 X_train, X_test, y_train, y_test = train_test_split(
#                     X_scaled, y, test_size=0.2, random_state=42
#                 )

#                 dt_model = DecisionTreeClassifier(
#                     max_depth=10,
#                     min_samples_leaf=2,
#                     min_samples_split=4,
#                     min_impurity_decrease=0.01,
#                     ccp_alpha=0.1,
#                     random_state=42
#                 )

#                 dt_model.fit(X_train, y_train)
#                 y_pred_dt = dt_model.predict(X_test)

#                 dt_metrics = classification_report(y_test, y_pred_dt, output_dict=True)
#                 acc = dt_metrics.get("accuracy", 0)
#                 precision = dt_metrics['weighted avg']['precision'] * 100
#                 recall = dt_metrics['weighted avg']['recall'] * 100
#                 f1 = dt_metrics['weighted avg']['f1-score'] * 100
#                 self.stdout.write(self.style.SUCCESS(f"✅ Decision Tree Accuracy: {acc * 100:.2f}%"))
                
#                 self.stdout.write(self.style.SUCCESS(f"✅ Precision: {precision:.2f}%"))
#                 self.stdout.write(self.style.SUCCESS(f"✅ Recall: {recall:.2f}%"))
#                 self.stdout.write(self.style.SUCCESS(f"✅ F1-score: {f1:.2f}%"))
#                 joblib.dump(dt_model, os.path.join(model_dir, 'dt_model.pkl'))
#                 joblib.dump(dt_metrics, os.path.join(model_dir, 'dt_metrics.pkl'))
#                 joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
#                 self.stdout.write(self.style.SUCCESS("✅ Decision Tree model, metrics, and scaler saved."))

#             except Exception as e:
#                 self.stdout.write(self.style.ERROR(f"❌ Error during Decision Tree training: {e}"))
#         else:
#             self.stdout.write(self.style.ERROR("❌ 'is_graduate' column missing."))

#         # --- SVM Academic Model ---
#         if 'is_graduate' in df.columns:
#             try:
#                 X = df.drop(columns=['is_graduate', 'id', 'department', 'feedback'], errors='ignore')
#                 if 'sentiments' not in X.columns and 'sentiments' in df.columns:
#                     X['sentiments'] = df['sentiments']
#                 y = df['is_graduate']

#                 svm_academic = make_pipeline(
#                     StandardScaler(),
#                     SVC(probability=True, kernel='rbf', C=1.0, gamma='scale')
#                 )

#                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#                 svm_academic.fit(X_train, y_train)
#                 y_pred_svm = svm_academic.predict(X_test)

#                 svm_academic_metrics = classification_report(y_test, y_pred_svm, output_dict=True)
                
#                 if svm_academic_metrics:
#                     acc = svm_academic_metrics.get("accuracy", 0) * 100
#                     prec = svm_academic_metrics["weighted avg"]["precision"] * 100
#                     rec = svm_academic_metrics["weighted avg"]["recall"] * 100
#                     f1 = svm_academic_metrics["weighted avg"]["f1-score"] * 100

#                     self.stdout.write(self.style.SUCCESS(f"📊 SVM Academic Accuracy: {acc:.2f}%"))
#                     self.stdout.write(self.style.SUCCESS(f"📊 SVM Academic Precision: {prec:.2f}%"))
#                     self.stdout.write(self.style.SUCCESS(f"📊 SVM Academic Recall: {rec:.2f}%"))
#                     self.stdout.write(self.style.SUCCESS(f"📊 SVM Academic F1-score: {f1:.2f}%"))

#                 joblib.dump(svm_academic, os.path.join(model_dir, 'svm_academic.pkl'))
#                 joblib.dump(svm_academic_metrics, os.path.join(model_dir, 'svm_academic_metrics.pkl'))
#                 # SVM Academic Model Metrics

#                 self.stdout.write(self.style.SUCCESS("✅ SVM Academic model and metrics saved."))

#             except Exception as e:
#                 self.stdout.write(self.style.ERROR(f"❌ SVM Academic model training failed: {e}"))

#         # --- SVM Sentiment Model ---
#         if 'feedback' in df.columns and 'sentiments' in df.columns:
#             try:
#                 X_text = df['feedback'].fillna("").astype(str)
#                 y_sent = df['sentiments']

#                 X_train, X_test, y_train, y_test = train_test_split(X_text, y_sent, test_size=0.2, random_state=42)

#                 sentiment_pipeline = make_pipeline(
#                     TfidfVectorizer(),
#                     SVC(probability=True, kernel='rbf', C=1.0, gamma='scale')
#                 )

#                 sentiment_pipeline.fit(X_train, y_train)
#                 y_pred = sentiment_pipeline.predict(X_test)

#                 sentiment_metrics = classification_report(y_test, y_pred, output_dict=True)
#                 if sentiment_metrics:
#                     acc = sentiment_metrics.get("accuracy", 0) * 100
#                     prec = sentiment_metrics["weighted avg"]["precision"] * 100
#                     rec = sentiment_metrics["weighted avg"]["recall"] * 100
#                     f1 = sentiment_metrics["weighted avg"]["f1-score"] * 100

#                     self.stdout.write(self.style.SUCCESS(f"📊 SVM Academic Accuracy: {acc:.2f}%"))
#                     self.stdout.write(self.style.SUCCESS(f"📊 SVM Academic Precision: {prec:.2f}%"))
#                     self.stdout.write(self.style.SUCCESS(f"📊 SVM Academic Recall: {rec:.2f}%"))
#                     self.stdout.write(self.style.SUCCESS(f"📊 SVM Academic F1-score: {f1:.2f}%"))
#                 joblib.dump(sentiment_pipeline, os.path.join(model_dir, 'svm_feedback.pkl'))
#                 joblib.dump(sentiment_metrics, os.path.join(model_dir, 'svm_metrics.pkl'))
#                 self.stdout.write(self.style.SUCCESS("✅ SVM Sentiment model and metrics saved."))

#             except Exception as e:
#                 self.stdout.write(self.style.ERROR(f"❌ SVM Sentiment model training failed: {e}"))

#         self.stdout.write(self.style.SUCCESS("🏁 Model training complete."))
