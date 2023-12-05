from django import forms
from .models import TextData

class InvoiceItemForm(forms.ModelForm):
    class Meta:
        model = TextData
        fields = ['input_text',]
        exclude = ['processed_text',]