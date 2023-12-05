from rest_framework import serializers
from .models import TextData

class DataSerializer(serializers.Serializer):
    input_text = serializers.CharField(max_length=10000)

    def create(self, validated_data):
        # Create and return a new instance of YourModel with the validated data
        return TextData.objects.create(**validated_data)
    processed_text = serializers.CharField(max_length=10000, required=False)