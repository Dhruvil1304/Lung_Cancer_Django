from django import forms


class ImageUploadForm(forms.Form):
    image = forms.ImageField()


class SymptomForm(forms.Form):
    GENDER_CHOICES = (
        ('M', 'Male'),
        ('F', 'Female'),
    )

    FIRST_NAME = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control'}))
    LAST_NAME = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control'}))
    EMAIL_ID = forms.EmailField(widget=forms.EmailInput(attrs={'class': 'form-control'}))
    GENDER = forms.ChoiceField(choices=GENDER_CHOICES, label='Gender', widget=forms.Select(attrs={'class': 'form'
                                                                                                           '-control'}))
    AGE = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    SMOKING = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    YELLOW_FINGERS = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    ANXIETY = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    PEER_PRESSURE = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    CHRONIC_DISEASE = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    FATIGUE = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    ALLERGY = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    WHEEZING = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    ALCOHOL_CONSUMING = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    COUGHING = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    SHORTNESS_OF_BREATH = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    SWALLOWING_DIFFICULTY = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    CHEST_PAIN = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
