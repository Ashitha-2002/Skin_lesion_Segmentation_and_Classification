from django.db import models
from django.utils import timezone
import os

def upload_to(instance, filename):
    return f'uploads/{timezone.now().strftime("%Y/%m/%d")}/{filename}'

class LesionAnalysis(models.Model):
    LESION_CLASSES = [
        ('MEL', 'Melanoma'),
        ('NV', 'Melanocytic nevus'),
        ('BCC', 'Basal cell carcinoma'),
        ('AK', 'Actinic keratosis'),
        ('BKL', 'Benign keratosis'),
        ('DF', 'Dermatofibroma'),
        ('VASC', 'Vascular lesion'),
        ('SCC', 'Squamous cell carcinoma'),
    ]
    
    image = models.ImageField(upload_to=upload_to, blank=True, null=True)
    segmentation_mask = models.ImageField(upload_to='masks/', blank=True, null=True)
    segmented_region = models.ImageField(upload_to='regions/', blank=True, null=True)
    predicted_class = models.CharField(max_length=4, choices=LESION_CLASSES, blank=True)
    confidence_score = models.FloatField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f'Analysis {self.id} - {self.get_predicted_class_display()}'