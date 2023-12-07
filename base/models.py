from django.db import models

class Chat(models.Model):
    user_message = models.CharField(max_length=10000, null=True, blank=True)
    bot_response = models.CharField(max_length=10000, null=True, blank=True)