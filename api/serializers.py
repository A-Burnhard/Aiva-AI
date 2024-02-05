from rest_framework import serializers
from .models import Chat

class DataSerializer(serializers.ModelSerializer):
    user_message = serializers.CharField(max_length=1000)

    class Meta:
        model = Chat
        fields = ('user_message',)
