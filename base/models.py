from django.db import models

class Chat(models.Model):
    user_message = models.CharField(max_length=1000, null=True, blank=True)
    bot_response = models.CharField(max_length=1000, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.user_message
    