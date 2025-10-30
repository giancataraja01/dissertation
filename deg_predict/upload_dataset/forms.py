from django import forms
from django.core.exceptions import ValidationError


class UploadCSVForm(forms.Form):
    file = forms.FileField(
        label='Upload CSV File',
        widget=forms.ClearableFileInput(attrs={
            'accept': '.csv',
            'class': 'form-control'
        }))


class UploadStudentCSVForm(forms.Form):
    file = forms.FileField(
        label="Upload CSV File",
        widget=forms.ClearableFileInput(attrs={
            'accept': '.csv',
            'class': 'form-control'
        }))


def clean_file(self):
    file = self.cleaned_data.get('file')
    if not file.name.endswith('.csv'):
        raise forms.ValidationError("❌ Please upload a valid CSV file.")
    return file
