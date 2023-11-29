from django.db import models

class FaceImage(models.Model):
    name = models.CharField(max_length=50)
    Image = models.ImageField(upload_to='images/')
    
    def __str__(self): 
	    return self.name
