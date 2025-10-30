from django.shortcuts import render, redirect
from django.http import HttpResponse
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
from upload_dataset.models import PredictionResult, UploadedStudent
import matplotlib
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib import colors
from datetime import datetime
from django.shortcuts import redirect
from django.http import HttpResponseForbidden
matplotlib.use('Agg')  # Use non-interactive backend

# Create your views here.

def m_report(request):
    print("role",request.session.get("role"))
    if request.session.get("role")==4:
        return HttpResponseForbidden("You are not allowed to access this page.")
    return render(request, "reports/main_report.html")

def graduation_chart(request):
    
    return render(request,"reports/graduate_report.html")

def hybrid_result_distribution_chart(request):
 #   from upload_dataset.models import PredictionResult

    results = PredictionResult.objects.values_list('code', flat=True)

    # 1. Count occurrences
    count_map = {}
    for code in results:
        count_map[code] = count_map.get(code, 0) + 1

    # if not count_map:
    #     return _render_error_image("No prediction results found.")

    # 2. Prepare sorted data
    total = sum(count_map.values())
    sorted_items = sorted(count_map.items(), key=lambda x: x[1], reverse=True)

    labels = []
    values = []
    percentages = []

    for code, count in sorted_items:
        percent = (count / total) * 100
        percentages.append(percent)
        labels.append(f"{code} - {percent:.1f}%")
        values.append(count)

    # 3. Plot
    plt.figure(figsize=(12, max(4, len(labels) * 0.6)))
    bars = plt.barh(labels, values, color='skyblue')

    # Add percentage labels to the bars
    for bar, value, percent in zip(bars, values, percentages):
        plt.text(value, bar.get_y() + bar.get_height()/2, f"{value} ({percent:.1f}%)",
                 va='center', fontsize=10)

    plt.xlabel("Number of Students")
    plt.title("Hybrid Weighted Result Prediction Distribution")
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    return HttpResponse(buffer.getvalue(), content_type='image/png')

def graduate_prediction_by_college(request):
    data = PredictionResult.objects.values('coll', 'code')
    
    result_map = {}  # {course: {code: count}}

    for entry in data:
        course = entry['coll']
        code = entry['code']
        result_map.setdefault(course, {})
        result_map[course][code] = result_map[course].get(code, 0) + 1

    # Flatten to plotting values
    courses = []
    likely = []
    moderately=[]
    unlikely = []

    for course, code_counts in result_map.items():
        courses.append(course)
        likely.append(code_counts.get("Likely to complete a degree", 0))
        moderately.append(code_counts.get("Moderately likely to complete a degree", 0))
        unlikely.append(code_counts.get("Unlikely to complete a degree", 0))

    # Plotting
    x = range(len(courses))
    plt.figure(figsize=(14, 6))
    plt.bar(x, likely, label="Likely", color='green')
    plt.bar(x, moderately, bottom=likely, label="Moderately Likely", color='orange')
    combined = [l + m for l, m in zip(likely, moderately)]
    plt.bar(x, unlikely, bottom=combined, label="Unlikely", color='red')

    for i in range(len(courses)):
        total = likely[i] + moderately[i] + unlikely[i]
        if total == 0:
            continue
        # Position at top of the bar stack
        top = combined[i] + unlikely[i] + 1
        label = (
            f"L: {likely[i]/total*100:.1f}%\n"
            f"M: {moderately[i]/total*100:.1f}%\n"
            f"U: {unlikely[i]/total*100:.1f}%"
        )
        plt.text(x[i], top, label, ha='center', fontsize=8)

    plt.xticks(x, courses, rotation=45, ha='right')
    plt.ylabel("Number of Students")
    plt.title("Graduate Prediction by College")
    plt.legend()
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    return HttpResponse(buffer.getvalue(), content_type='image/png')

def graduate_prediction_by_course(request):
    data = PredictionResult.objects.values('course', 'code')
    
    result_map = {}  # {course: {code: count}}

    for entry in data:
        course = entry['course']
        code = entry['code']
        result_map.setdefault(course, {})
        result_map[course][code] = result_map[course].get(code, 0) + 1

    # Flatten to plotting values
    courses = []
    likely = []
    moderately=[]
    unlikely = []

    for course, code_counts in result_map.items():
        courses.append(course)
        likely.append(code_counts.get("Likely to complete a degree", 0))
        moderately.append(code_counts.get("Moderately likely to complete a degree", 0))
        unlikely.append(code_counts.get("Unlikely to complete a degree", 0))

    # Plotting
    x = range(len(courses))
    plt.figure(figsize=(14, 6))
    plt.bar(x, likely, label="Likely", color='green')
    plt.bar(x, moderately, bottom=likely, label="Moderately Likely", color='orange')
    combined = [l + m for l, m in zip(likely, moderately)]
    plt.bar(x, unlikely, bottom=combined, label="Unlikely", color='red')

    for i in range(len(courses)):
        total = likely[i] + moderately[i] + unlikely[i]
        if total == 0:
            continue
        # Position at top of the bar stack
        top = combined[i] + unlikely[i] + 1
        label = (
            f"L: {likely[i]/total*100:.1f}%\n"
            f"M: {moderately[i]/total*100:.1f}%\n"
            f"U: {unlikely[i]/total*100:.1f}%"
        )
        plt.text(x[i], top, label, ha='center', fontsize=8)

    plt.xticks(x, courses, rotation=45, ha='right')
    plt.ylabel("Number of Students")
    plt.title("Graduate Prediction by Course")
    plt.legend()
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    return HttpResponse(buffer.getvalue(), content_type='image/png')


def at_risk_students_prediction_by_course(request):
    
     # Get only "Unlikely" predictions grouped by course
    data = PredictionResult.objects.filter(
        code="Unlikely to complete a degree"
    ).values_list('course', flat=True)

    course_counts = {}
    for course in data:
        course_counts[course] = course_counts.get(course, 0) + 1

    if not course_counts:
        return _render_error_image("No at-risk data found.")

    # Sort and get top 10 courses by number of at-risk students
    sorted_courses = sorted(course_counts.items(), key=lambda x: x[1], reverse=True)
    top_courses = sorted_courses[:10]
    labels = [f"{course} ({count})" for course, count in top_courses]
    values = [count for _, count in top_courses]

    # Create pie chart
    plt.figure(figsize=(10, 7))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Distribution of At-Risk Students by Course (Top 10)")
    plt.axis('equal')
    plt.tight_layout()

    # Return chart as image response
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    return HttpResponse(buffer.getvalue(), content_type='image/png')


def _render_error_image(message):
    plt.figure(figsize=(8, 4))
    plt.text(0.5, 0.5, message, ha='center', va='center', fontsize=12)
    plt.axis('off')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    return HttpResponse(buffer.getvalue(), content_type='image/png')


def at_risk_students_by_name(request):
    # Get IDs of students predicted unlikely to complete
    at_risk = PredictionResult.objects.filter(code="Unlikely to complete a degree")

    # Join with UploadedStudent using stud_id = user_id
    students = UploadedStudent.objects.filter(
        user_id__in=[r.stud_id for r in at_risk]
    )

    # Map user_id to prediction entry for course info
    prediction_map = {r.stud_id: r for r in at_risk}

    # Group by course
    grouped = {}
    for student in students:
        course = prediction_map.get(student.user_id).course if student.user_id in prediction_map else "Unknown"
        full_name = f"{student.lastname}, {student.firstname} {student.mi or ''}".strip()
        grouped.setdefault(course, []).append(full_name)

    # Sort the courses and student names
    grouped_sorted = {course: sorted(names) for course, names in sorted(grouped.items())}

    return render(request, "reports/at_risk_students.html", {
        "grouped_students": grouped_sorted
    })

def download_at_risk_students_pdf(request):
     # Get students predicted as "Unlikely to complete degree"
    at_risk = PredictionResult.objects.filter(code="Unlikely to complete degree")
    prediction_map = {r.stud_id: r.course for r in at_risk}
    students = UploadedStudent.objects.filter(user_id__in=prediction_map.keys())

    # Group full names by course
    grouped = {}
    for student in students:
        course = prediction_map.get(student.user_id, "Unknown")
        full_name = f"{student.lastname}, {student.firstname} {student.mi or ''}".strip()
        grouped.setdefault(course, []).append(full_name)

    # Prepare PDF response
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="At_Risk_Students_By_Course.pdf"'

    p = canvas.Canvas(response, pagesize=letter)
    width, height = letter

    def draw_header(y):
        headers = [
            ("University of Cebu - Main Campus", "Helvetica-Bold", 12),
            ("Office of the Vice-Chancellor for Academics", "Helvetica", 11),
            ("At-Risk Students by Course - Report", "Helvetica-Bold", 11)
        ]
        for text, font, size in headers:
            p.setFont(font, size)
            text_width = stringWidth(text, font, size)
            p.drawString((width - text_width) / 2, y, text)
            y -= 18
        return y - 10  # Additional spacing

    def draw_footer(page_num):
        p.setFont("Helvetica", 8)
        p.drawString(50, 30, f"Date Printed: {datetime.now().strftime('%B %d, %Y')}")
        p.drawRightString(width - 50, 30, f"Page {page_num}")

    y = height - 50
    page_num = 1
    y = draw_header(y)

    for course, names in sorted(grouped.items()):
        box_height = 30 + len(names) * 14 + 10  # 14px per name + padding
        if y - box_height < 60:
            draw_footer(page_num)
            p.showPage()
            page_num += 1
            y = height - 50
            y = draw_header(y)
            p.setFont("Helvetica", 9)

        # Draw border box
        box_top = y
        box_bottom = y - box_height
        p.setStrokeColor(colors.grey)
        p.rect(45, box_bottom, width - 90, box_height, stroke=1, fill=0)

        # Draw colored course header background
        p.setFillColor(colors.lightblue)
        p.rect(45, y - 20, width - 90, 20, fill=1, stroke=0)
        p.setFillColor(colors.black)

        # Course name
        p.setFont("Helvetica-Bold", 10)
        p.drawString(50, y - 15, f"Course: {course}")
        y -= 35

        # Student list
        p.setFont("Helvetica", 8)
        for idx, name in enumerate(names, start=1):
            if y < 60:
                draw_footer(page_num)
                p.showPage()
                page_num += 1
                y = height - 50
                y = draw_header(y)
                p.setFont("Helvetica", 8)
            p.drawString(70, y, f"{idx}. {name}")
            y -= 14

        y -= 20  # Extra space between course blocks

    draw_footer(page_num)
    p.save()
    return response

def sentiment_distribution_chart(request):
    # Step 1: Fetch sentiment values
    sentiments = PredictionResult.objects.values_list('svm_prediction', flat=True)

    # Step 2: Count each category
    count_map = {
        'Positive': 0,
        'Negative': 0,
        'Neutral': 0
    }
    for s in sentiments:
        if s in count_map:
            count_map[s] += 1

    # Step 3: Filter out sentiments with 0 count
    labels = []
    values = []
    colors = []

    for sentiment, count in count_map.items():
        if count > 0:
            labels.append(f"{sentiment} ({count})")
            values.append(count)
            colors.append({
                'Positive': 'green',
                'Negative': 'red',
                'Neutral': 'orange'
            }[sentiment])

    # Step 4: Generate pie chart
    plt.figure(figsize=(10, 7))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title("Student Sentiments Distribution")
    plt.axis('equal')  # Make it circular

    # Step 5: Return as image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    return HttpResponse(buffer.getvalue(), content_type='image/png')

def sentiment_wordcloud_chart(request):        
    # Step 1: Get all sentiments
        sentiments = PredictionResult.objects.values_list('feedback', flat=True)

        # Step 2: Count frequency
        freq = {}
        for s in sentiments:
            s = s.strip().capitalize()
            freq[s] = freq.get(s, 0) + 1

        # Step 3: Generate word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='Set2'
        ).generate_from_frequencies(freq)

        # Step 4: Render image
        buffer = BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)

        return HttpResponse(buffer.getvalue(), content_type='image/png')
