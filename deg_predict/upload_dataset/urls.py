from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_dataset, name='upload_dataset'),
    path('upload-students/', views.upload_student, name='upload_student'),
    path('risk-report/', views.risk_report_chart, name='risk_report_chart'),
    # path('model-performance-chart/', views.model_performance_chart, name='model_performance_chart'),
    
]
