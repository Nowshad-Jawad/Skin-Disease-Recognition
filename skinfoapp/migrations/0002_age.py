# Generated by Django 3.1.6 on 2021-02-21 20:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('skinfoapp', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Age',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('age', models.CharField(max_length=10)),
            ],
        ),
    ]
