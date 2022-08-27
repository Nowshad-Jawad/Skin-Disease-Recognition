# Generated by Django 3.1.6 on 2021-02-21 21:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('skinfoapp', '0002_age'),
    ]

    operations = [
        migrations.DeleteModel(
            name='Age',
        ),
        migrations.AddField(
            model_name='image',
            name='age',
            field=models.IntegerField(null=True),
        ),
        migrations.AddField(
            model_name='image',
            name='area',
            field=models.CharField(choices=[('S', 'Scalp'), ('E', 'Ear'), ('F', 'Face'), ('B', 'Back'), ('T', 'Trunk'), ('C', 'Chest'), ('U', 'Upper-Extremity'), ('L', 'Lower-Extremity'), ('A', 'Abdomen'), ('G', 'Genital'), ('H', 'Hand'), ('N', 'Neck')], max_length=20, null=True),
        ),
        migrations.AddField(
            model_name='image',
            name='gender',
            field=models.CharField(choices=[('M', 'Male'), ('F', 'Female'), ('O', 'Others')], max_length=7, null=True),
        ),
    ]
