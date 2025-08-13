from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.core.files.base import ContentFile
from .models import LesionAnalysis
from .forms import ImageUploadForm
from .ml_utils import LesionClassifier
import io

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                analysis = form.save()
                classifier = LesionClassifier()
                predicted_class, confidence = classifier.classify_lesion(analysis.image.path)
                analysis.predicted_class = predicted_class
                analysis.confidence_score = confidence
                mask_image, segmented_image = classifier.generate_segmentation_mask(analysis.image.path)

                if mask_image and segmented_image:
                    mask_io = io.BytesIO()
                    mask_image.save(mask_io, format='PNG')
                    mask_content = ContentFile(mask_io.getvalue())
                    analysis.segmentation_mask.save(f'mask_{analysis.id}.png', mask_content, save=False)

                    segmented_io = io.BytesIO()
                    segmented_image.save(segmented_io, format='PNG')
                    segmented_content = ContentFile(segmented_io.getvalue())
                    analysis.segmented_region.save(f'segmented_{analysis.id}.png', segmented_content, save=False)

                analysis.save()
                messages.success(request, 'Image analyzed successfully!')
                return redirect('lesion_analyzer:results', analysis_id=analysis.id)
            except Exception as e:
                messages.error(request, f'Error analyzing image: {str(e)}')
        else:
            messages.error(request, 'Please upload a valid image file.')
    else:
        form = ImageUploadForm()
    return render(request, 'lesion_analyzer/upload.html', {'form': form})

def view_results(request, analysis_id):
    analysis = get_object_or_404(LesionAnalysis, id=analysis_id)
    return render(request, 'lesion_analyzer/results.html', {'analysis': analysis})

def analysis_history(request):
    analyses = LesionAnalysis.objects.all()[:20]
    return render(request, 'lesion_analyzer/history.html', {'analyses': analyses})
