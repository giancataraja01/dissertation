from django.db import models

# Create your models here.

from django.core.exceptions import ValidationError


class StudentRecord(models.Model):
    shs_gpa = models.FloatField()
    coll_gpa = models.FloatField()
    age = models.IntegerField()
    sex = models.IntegerField()
    is_boarding = models.IntegerField()
    is_living_with_family = models.IntegerField()
    is_province = models.IntegerField()
    is_scholar = models.IntegerField()
    father_educ = models.IntegerField()
    mother_educ = models.IntegerField()
    financial_status = models.IntegerField()
    is_graduate = models.TextField(blank=True, null=True)
    sentiments = models.IntegerField()
    department = models.TextField(blank=True, null=True)
    feedback = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"Student (Age: {self.age}, SHS GPA: {self.shs_gpa})"


class UploadedStudent(models.Model):
    user_id = models.CharField(
        max_length=20,
        unique=True,
    )
    username = models.CharField(max_length=100)
    # For security, use hashed passwords
    password = models.CharField(max_length=100)
    lastname = models.CharField(max_length=100)
    firstname = models.CharField(max_length=100)
    mi = models.CharField(max_length=10, blank=True, null=True)
    course = models.CharField(max_length=100)
    is_updated = models.BooleanField(default=False)
    role = models.IntegerField()
    coll = models.CharField(max_length=10, blank=True, null=True)
    # feedback = models.TextField()


    # def clean(self):
    #     if self.coll and len(self.coll) > 10:
    #         raise ValidationError({'coll': 'College abbreviation must be 10 characters or less'})

    def __str__(self):
        return self.username


class RiskIndicator(models.Model):
    student_id = models.CharField(max_length=20)
    risk_message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Risk for {self.student.user_id}: {self.risk_message[:30]}..."


class PredictionResult(models.Model):
   # student = models.ForeignKey('StudentRecord', on_delete=models.CASCADE, related_name='predictions')
    stud_id = models.CharField(max_length=20)
    coll = models.CharField(max_length=10,null=True)

    # dt_prediction = models.CharField(max_length=100)
    # dt_confidence = models.FloatField()
    
    svm_prediction = models.CharField(max_length=100)
    # svm_confidence = models.FloatField()
    
    hybrid_result = models.TextField()  # full sentence explanation
    hybrid_score = models.FloatField()
    
    average_result = models.TextField() 
    average_score = models.FloatField()
    code = models.TextField()
    course = models.TextField()
    feedback = models.TextField()

    average_score = models.FloatField(default=0.0)
    # predicted_at = models.DateTimeField(auto_now_add=True)

    # def __str__(self):
    #     return f"Prediction for Student {self.student.user_id} at {self.predicted_at.strftime('%Y-%m-%d %H:%M')}"
