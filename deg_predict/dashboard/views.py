from django.shortcuts import render, redirect
from django.conf import settings
from django.views.decorators.cache import cache_control
import os
import joblib

def _load_pkl(path):
    return joblib.load(path) if os.path.exists(path) else None


@cache_control(no_cache=True, must_revalidate=True, no_store=True)

def dash(request):

    # if 'user_id' not in request.session:
    #     return redirect('login')
    if not request.session.get("role"):
        return redirect('login')
    
    ml_dir = os.path.join(settings.BASE_DIR, 'predict', 'ml')

    # Base model metrics
    dt_metrics = _load_pkl(os.path.join(ml_dir, 'dt_metrics.pkl'))
    svm_metrics = _load_pkl(os.path.join(ml_dir, 'svm_academic_metrics.pkl'))

    # Hybrid metrics (true ensemble results from train_model.py)
    soft_metrics  = _load_pkl(os.path.join(ml_dir, 'hybrid_soft_metrics.pkl'))
    hard_metrics  = _load_pkl(os.path.join(ml_dir, 'hybrid_hard_metrics.pkl'))
    stack_metrics = _load_pkl(os.path.join(ml_dir, 'hybrid_stack_metrics.pkl'))

    # Helper to safely extract metric fields (values are already % in our saved dicts)
    def get(m, key, default=None):
        return m.get(key) if (m and key in m) else default

    context = {
        # --- Decision Tree ---
        "dt_acc":  get(dt_metrics, 'accuracy', 0) * 100,
        "dt_prec": get(dt_metrics, 'weighted avg', {}).get('precision', 0) * 100 if dt_metrics else 0,
        "dt_rec":  get(dt_metrics, 'weighted avg', {}).get('recall', 0) * 100 if dt_metrics else 0,
        "dt_f1":   get(dt_metrics, 'weighted avg', {}).get('f1-score', 0) * 100 if dt_metrics else 0,

        # --- SVM Academic ---
        "svm_acc":  get(svm_metrics, 'accuracy', 0) * 100,
        "svm_prec": get(svm_metrics, 'weighted avg', {}).get('precision', 0) * 100 if svm_metrics else 0,
        "svm_rec":  get(svm_metrics, 'weighted avg', {}).get('recall', 0) * 100 if svm_metrics else 0,
        "svm_f1":   get(svm_metrics, 'weighted avg', {}).get('f1-score', 0) * 100 if svm_metrics else 0,

        # --- True Hybrid metrics (already saved as percentages in train_model.py) ---
        # SOFT
        "hybrid_soft_acc":  get(soft_metrics, 'accuracy'),
        "hybrid_soft_prec": get(soft_metrics, 'precision'),
        "hybrid_soft_rec":  get(soft_metrics, 'recall'),
        "hybrid_soft_f1":   get(soft_metrics, 'f1'),

        # HARD
        "hybrid_hard_acc":  get(hard_metrics, 'accuracy'),
        "hybrid_hard_prec": get(hard_metrics, 'precision'),
        "hybrid_hard_rec":  get(hard_metrics, 'recall'),
        "hybrid_hard_f1":   get(hard_metrics, 'f1'),

        # STACK
        "hybrid_stack_acc":  get(stack_metrics, 'accuracy'),
        "hybrid_stack_prec": get(stack_metrics, 'precision'),
        "hybrid_stack_rec":  get(stack_metrics, 'recall'),
        "hybrid_stack_f1":   get(stack_metrics, 'f1'),
    }

    return render(request, "dashboard/dashboard.html", context)



##########################################################################
# from django.shortcuts import render, redirect
# from django.contrib.auth.decorators import login_required
# import os
# import joblib
# from django.conf import settings
# from django.shortcuts import render
# import matplotlib.pyplot as plt

# # Create your views here.


# #@login_required
# def dash(request):
#     if 'user_id' not in request.session:  
#         return redirect('login')
#     dt_metrics_path = os.path.join(settings.BASE_DIR, 'predict/ml/dt_metrics.pkl')
#     svm_metrics_path = os.path.join(settings.BASE_DIR, 'predict/ml/svm_academic_metrics.pkl')
#     if os.path.exists(dt_metrics_path):
#                 dt_metrics = joblib.load(dt_metrics_path)
#     if os.path.exists(svm_metrics_path):
#                 svm_metrics = joblib.load(svm_metrics_path)

#     return render(request, "dashboard/dashboard.html", {
#           "dt_acc": dt_metrics['accuracy'] * 100,
#           "dt_prec": dt_metrics['weighted avg']['precision'] * 100,
#           "dt_rec": dt_metrics['weighted avg']['recall'] * 100,
#           "dt_f1": dt_metrics['weighted avg']['f1-score'] * 100,

#           "svm_acc": svm_metrics['accuracy'] * 100,
#           "svm_prec": svm_metrics['weighted avg']['precision'] * 100,
#           "svm_rec": svm_metrics['weighted avg']['recall'] * 100,
#           "svm_f1": svm_metrics['weighted avg']['f1-score'] * 100,

#           "hybrid_acc_ave": (dt_metrics['accuracy'] + svm_metrics['accuracy']) / 2 * 100,
#           "hybrid_acc_wei": (dt_metrics['accuracy'] *.8 + svm_metrics['accuracy']*.2) * 100,
                
#           "hybrid_prec_ave":(dt_metrics['weighted avg']['precision'] + svm_metrics['weighted avg']['precision']) / 2 * 100,
#           "hybrid_prec_wei":(dt_metrics['weighted avg']['precision'] *.8 + svm_metrics['weighted avg']['precision'] * .2)* 100,
                
#           "hybrid_rec_ave": (dt_metrics['weighted avg']['recall'] + svm_metrics['weighted avg']['recall']) /2 * 100,
#           "hybrid_rec_wei": (dt_metrics['weighted avg']['recall']*.8 + svm_metrics['weighted avg']['recall'] *.2) * 100,

#           "hybrid_f1_ave": (dt_metrics['weighted avg']['f1-score'] + svm_metrics['weighted avg']['f1-score']) / 2 * 100,
#           "hybrid_f1_wei": (dt_metrics['weighted avg']['f1-score']*.8 + svm_metrics['weighted avg']['f1-score'] *.2) * 100,
    
#     })

