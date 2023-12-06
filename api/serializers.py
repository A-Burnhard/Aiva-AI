from rest_framework import serializers
from .models import TextData

class DataSerializer(serializers.ModelSerializer):
    input_text = serializers.CharField(max_length=10000)

    class Meta:
        model = TextData
        fields = ('input_text', 'processed_text')