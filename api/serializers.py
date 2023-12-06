from rest_framework import serializers
from base.models import Chat

class DataSerializer(serializers.ModelSerializer):
    user_message = serializers.CharField(max_length=10000)
    bot_response = serializers.CharField(max_length=10000)


    class Meta:
        model = Chat
        fields = ('user_message', 'bot_response')