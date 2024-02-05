from django.db import models

class Chat(models.Model):
    user_message = models.CharField(max_length=10000, blank=True, null=True)

    def __str__(self):
        return self.user_message