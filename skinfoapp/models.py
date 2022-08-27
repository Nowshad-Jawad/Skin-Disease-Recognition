from django.db import models

# Create your models here.
class Image(models.Model):
    Age = models.IntegerField(null=True)
    GENDER = (('M', 'Male'),
            ('F', 'Female'),
            ('O','Others'))
    Gender = models.CharField(max_length=7, choices = GENDER, null=True)
    LOCALIZATION = (('S','Scalp'), ('E', 'Ear'),
    ('F', 'Face'), ('B', 'Back'), ('T', 'Trunk'), ('C', 'Chest'),
     ('U', 'Upper-Extremity'), ('L', 'Lower-Extremity'), ('A', 'Abdomen'),
     ('G', 'Genital'), ('H', 'Hand'), ('N', 'Neck'))
    Affected_Area = models.CharField(max_length=20, choices = LOCALIZATION, null=True)
    photo = models.ImageField(upload_to = "media/")
    date = models.DateTimeField(auto_now_add=True)
