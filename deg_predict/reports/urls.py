from django.urls import path
from . import views
from upload_dataset import views as upload_dataset_views

urlpatterns = [
#     path("", views.m_report,name="m_report"),
#    # path('reports/charts/hybrid-distribution/', views.hybrid_result_distribution_chart, name='hybrid_distribution_chart'),
#      path("graduate/", views.graduation_chart, name="graduation_chart"),
    
#     # View that returns the chart image (PNG)
#     path("charts/hybrid-distribution/", views.hybrid_result_distribution_chart, name="hybrid_distribution_chart"),
path('', views.m_report, name='m_report'),
path('charts/hybrid-distribution/', views.hybrid_result_distribution_chart, name='hybrid_distribution_chart'),
#path('charts/risk-report/', upload_dataset_views.risk_report_chart, name='risk_report_chart'),
path('charts/risk-report/', upload_dataset_views.risk_report_chart, {'chart_type': 'bar'}, name='risk_report_bar'),
path('charts/risk-report-pie/', upload_dataset_views.risk_report_chart, {'chart_type': 'pie'}, name='risk_report_pie'),
path('charts/graduate-by-college/', views.graduate_prediction_by_college, name='graduate_by_college'),
path('charts/graduate-by-course/', views.graduate_prediction_by_course, name='graduate_by_course'),
path('charts/at-risk-students-by-course/', views.at_risk_students_prediction_by_course, name='at_risk_students_by_course'),
#path('charts/at-risk-students/', views.at_risk_students_by_name, name='at_risk_students'),
path('charts/at-risk-students/', views.at_risk_students_by_name, name='at_risk_students'),
path('download/at-risk-students/pdf/', views.download_at_risk_students_pdf, name='download_at_risk_students_pdf'),
path('charts/sentiment-distribution/', views.sentiment_distribution_chart, name='sentiment_distribution'),
path('charts/sentiment-wordcloud/', views.sentiment_wordcloud_chart, name='sentiment_wordcloud'),


]

