import pandas as pd
from django.shortcuts import render, redirect
from .forms import UploadCSVForm
from .models import StudentRecord, UploadedStudent
from django.contrib import messages
import matplotlib.pyplot as plt
from django.http import HttpResponse
from io import BytesIO
from .models import RiskIndicator
import os
import joblib
from django.conf import settings
from django.shortcuts import render
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib in web apps
from django.views.decorators.cache import never_cache
from django.utils.decorators import method_decorator




def upload_dataset(request):
    if request.method == "POST":
        form = UploadCSVForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                csv_file = request.FILES['file']
                df = pd.read_csv(csv_file)

                # Optional: Rename columns to match model fields if needed
                df.columns = [col.strip().lower() for col in df.columns]
                duplicates = 0
                inserted = 0
                for _, row in df.iterrows():
                    
                    StudentRecord.objects.create(
                        shs_gpa=row['shs_gpa'],
                        coll_gpa=row['coll_gpa'],
                        age=row['age'],
                        sex=row['sex'],
                        is_boarding=row['is_boarding'],
                        is_living_with_family=row['is_living_with_family'],
                        is_province=row['is_province'],
                        is_scholar=row['is_scholar'],
                        father_educ=row['father_educ'],
                        mother_educ=row['mother_educ'],
                        financial_status=row['financial_status'],
                        is_graduate=row['is_graduate'],
                        feedback=row['feedback'],
                        sentiments=row['sentiments'],
                        department=row['department']
                    )
                    inserted += 1
                messages.success(request, f"✅ Uploaded: {inserted} student(s), Skipped: {duplicates} duplicate(s).")
                return redirect('upload_dataset')
                #messages.success(request, "Upload successful!")
                #return redirect('upload_dataset/upload_form.html')
               # return render(request, 'upload_dataset/upload_success.html', {
                #    'err_mess': "Only CSV files are accepted!",
               # })
            except Exception as e:
              messages.error(request, f"❌ Error: {str(e)}")
              return redirect('upload_student')
               
    else:
        form = UploadCSVForm()

    return render(request, 'upload_dataset/upload_form.html', {
        'form': form,
    })


def upload_student(request):
    if request.method == "POST":
        form = UploadCSVForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['file']
            try:
                # df = pd.read_csv(csv_file)
                # df.columns = [col.strip().lower() for col in df.columns]
                df = pd.read_csv(csv_file, encoding='ISO-8859-1')
                df.columns = [col.strip().lower() for col in df.columns]
                duplicates = 0
                inserted = 0

                for _, row in df.iterrows():
                    student_id = str(row['user_id'])
                    if UploadedStudent.objects.filter(user_id=student_id).exists():
                        duplicates += 1
                        continue  # Skip duplicate

                    UploadedStudent.objects.create(
                        user_id=row['user_id'],
                        username=row['username'],
                        password=row['password'],
                        lastname=row['lastname'],
                        firstname=row['firstname'],
                        mi=row['mi'],
                        course=row['course'],
                        is_updated=False,
                        role=row['role'],
                        coll = row['coll']
                    )
                    inserted += 1

                messages.success(
                    request, f"✅ Uploaded: {inserted} student(s), Skipped: {duplicates} duplicate(s).")
                # or your desired redirect target
                return redirect('upload_student')

            except Exception as e:
                messages.error(request, f"❌ Error: {str(e)}")
                return redirect('upload_student')
    else:
        form = UploadCSVForm()
    return render(request, 'upload_dataset/upload_students.html', {'form': form})


# def risk_report_chart(request):
#     # Count occurrences of each risk
#     risks = RiskIndicator.objects.values_list('risk_message', flat=True)
#     risk_count = {}

#     for risk in risks:
#         risk_count[risk] = risk_count.get(risk, 0) + 1

#     if not risk_count:
#         # Handle empty case
#         return _render_error_image("No risk indicators found.")

#     # Sort and prepare labels/values
#     sorted_risks = sorted(risk_count.items(), key=lambda x: x[1], reverse=True)
#     labels = [item[0] for item in sorted_risks]
#     values = [item[1] for item in sorted_risks]

#     # Plot
#     #plt.figure(figsize=(10, 5))
#     plt.figure(figsize=(12, max(4, len(labels) * 0.6)))
#     plt.barh(labels, values)
#     plt.xlabel("Number of Students")
#     plt.title("Risk Indicators Report")
#     plt.tight_layout()

#     buffer = BytesIO()
#     plt.savefig(buffer, format='png')
#     plt.close()
#     buffer.seek(0)

#     return HttpResponse(buffer.getvalue(), content_type='image/png')

def risk_report_chart(request, *args, **kwargs):
    chart_type = kwargs.get("chart_type", "bar")  # <- pull from path
    risks = RiskIndicator.objects.values_list('risk_message', flat=True)

    risk_count = {}
    for risk in risks:
        risk_count[risk] = risk_count.get(risk, 0) + 1

    if not risk_count:
        return _render_error_image("No risk indicators found.")

    sorted_risks = sorted(risk_count.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_risks]
    values = [item[1] for item in sorted_risks]

    plt.figure(figsize=(12, max(4, len(labels) * 0.6)))

    if chart_type == "pie":
        labels = labels[:10]
        values = values[:10]
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        plt.title("Risk Indicators Distribution (Pie Chart)")
    else:
        plt.barh(labels, values)
        plt.xlabel("Number of Students")
        plt.title("Risk Indicators Report (Bar Chart)")
        plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    response = HttpResponse(buffer.getvalue(), content_type='image/png')
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    return response
    return HttpResponse(buffer.getvalue(), content_type='image/png')


# Optional: Render fallback image on error
def _render_error_image(message):
    plt.figure(figsize=(10, 4))
    plt.text(0.5, 0.5, message, ha='center', va='center', wrap=True, fontsize=12)
    plt.axis('off')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    return HttpResponse(buffer.getvalue(), content_type='image/png')