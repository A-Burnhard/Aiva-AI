from django import forms
from .models import Chat

class Chat(forms.ModelForm):
    class Meta:
        model = Chat
        fields = ['user_message',]
        exclude = ['bot_response',]
