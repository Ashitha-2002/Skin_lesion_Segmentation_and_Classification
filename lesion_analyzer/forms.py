from django import forms
from .models import LesionAnalysis

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = LesionAnalysis
        fields = ['image']
        widgets = {
            'image': forms.FileInput(attrs={
                'class': 'form-control-file',
                'accept': 'image/*',
                'required': True,
                'id': 'imageInput'
            })
        }
    
    def clean_image(self):
        image = self.cleaned_data.get('image')
        if image:
            if image.size > 10 * 1024 * 1024:  # 10MB limit
                raise forms.ValidationError("Image file too large. Maximum size is 10MB.")
            if not image.content_type.startswith('image/'):
                raise forms.ValidationError("File must be an image.")
        return image