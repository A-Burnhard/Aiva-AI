from django.db import models

class TextData(models.Model):
    input_text = models.CharField(max_length=10000)
    processed_text = models.CharField(max_length=10000, null=True, blank=True)
