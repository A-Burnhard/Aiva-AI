# Generated by Django 5.0.1 on 2024-02-05 11:00

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('api', '0002_delete_textdata'),
    ]

    operations = [
        migrations.CreateModel(
            name='Chat',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user_message', models.CharField(blank=True, max_length=10000, null=True)),
                ('bot_response', models.CharField(blank=True, max_length=10000, null=True)),
            ],
        ),
    ]
