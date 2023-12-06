# Generated by Django 4.0.7 on 2023-12-06 12:50

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Chat',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user_message', models.CharField(max_length=10000)),
                ('bot_response', models.CharField(blank=True, max_length=10000, null=True)),
            ],
        ),
    ]