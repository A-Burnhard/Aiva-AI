from django.db import models

class Chat(models.Model):
    user_message = models.CharField(max_length=10000, blank=True, null=True)
    bot_response = models.CharField(max_length=10000, blank=True, null=True)

    def __str__(self):
        return f"Chat(id={self.id}, user_message={self.user_message}, bot_response={self.bot_response})"
