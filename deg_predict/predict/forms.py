from django import forms


class PredictorForm(forms.Form):
    shs_gpa = forms.FloatField(
        label="SHS GPA",
        min_value=1.0,
        max_value=5.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}))
    coll_gpa = forms.FloatField(
        label="College GPA Four Semesters",
        min_value=1.0,
        max_value=5.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}))

    age = forms.IntegerField(
        label="Age",
        min_value=1,
        max_value=100,
        widget=forms.NumberInput(attrs={'class': 'form-control'}))

    sex = forms.ChoiceField(
        choices=[(1, "Male"),
                 (0, "Female")],
        label="Sex",
        widget=forms.Select(attrs={'class': 'form-control'}))

    is_boarding = forms.ChoiceField(
        choices=[(1, "Yes"),
                 (0, "No")],
        label="Living in a Boarding House",
        widget=forms.Select(attrs={'class': 'form-control'}))

    is_liv_fam = forms.ChoiceField(
        choices=[(1, "Yes"),
                 (0, "No")],
        label="Living with Family",
        widget=forms.Select(attrs={'class': 'form-control'}))

    is_province = forms.ChoiceField(
        choices=[(1, "Yes"),
                 (0, "No")],
        label="Living in the Province",
        widget=forms.Select(attrs={'class': 'form-control'}))

    is_scholar = forms.ChoiceField(
        choices=[(1, "Yes"),
                 (0, "No")],
        label="Are you a scholar",
        widget=forms.Select(attrs={'class': 'form-control'}))

    father_educ = forms.ChoiceField(
        choices=[(1, "Elementary Education"),
                 (2, "Secondary Education"),
                 (3, "College Level"),
                 (4, "College Degree"),
                 (5, "Post Graduate")],
        label="Father Educational Background",
        widget=forms.Select(attrs={'class': 'form-control'}))

    mother_educ = forms.ChoiceField(
        choices=[(1, "Elementary Education"),
                 (2, "Secondary Education"),
                 (3, "College Level"),
                 (4, "College Degree"),
                 (5, "Post Graduate")],
        label="Mother Educational Background",
        widget=forms.Select(attrs={'class': 'form-control'}))

    financial_status = forms.ChoiceField(
        choices=[(1, "less than 10,957"),
                 (2, "10,957 to 21,914"),
                 (3, "21,915 to 43,828"),
                 (4, "43,829 to 76,669"),
                 (5, "76,670 to 131,484"),
                 (6, "131,485 to 219,140"),
                 (7, "219,141 and above")],
        label="Parents Financial Monthly Income",
        widget=forms.Select(attrs={'class': 'form-control'}))

    feedback = forms.CharField(
        label="Feedback/Comment",
        max_length=1050,
        widget=forms.Textarea(attrs={'class': 'form-control'}), required=True)


def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    for field in self.fields.values():
        field.widget.attrs['class'] = 'form-control'
