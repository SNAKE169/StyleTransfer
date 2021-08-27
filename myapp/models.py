from django.db import models
 
# Create your models here.
class StyleTransferModel(models.Model):
    img = models.ImageField()
    style = models.ImageField()