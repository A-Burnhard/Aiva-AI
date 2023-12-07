from rest_framework import serializers
from base.models import Chat

class DataSerializer(serializers.ModelSerializer):
    user_message = serializers.CharField(max_length=10000, required=False)
    bot_response = serializers.CharField(max_length=10000, required=False)


    class Meta:
        model = Chat
        fields = ('user_message', 'bot_response')
