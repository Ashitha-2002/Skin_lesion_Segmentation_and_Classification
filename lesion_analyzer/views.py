from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.core.files.base import ContentFile
from .models import LesionAnalysis
from .forms import ImageUploadForm
from .ml_utils import LesionClassifier
import io
from .models import LesionAnalysis  


def home(request):
    """Home page with introduction"""
    recent_analyses = LesionAnalysis.objects.all()[:6]  # Show 6 recent analyses
    context = {
        'recent_analyses': recent_analyses,
        'total_analyses': LesionAnalysis.objects.count()
    }
    return render(request, 'lesion_analyzer/home.html', context)

def lesion_types(request):
    lesion_data = [
        {
            'code': 'MEL',
            'name': 'Melanoma',
            'description': 'A serious form of skin cancer that develops in melanocytes, the cells that produce pigment.',
            'image_filename': 'melanoma_example.jpg',
            'characteristics': [
                'Asymmetrical shape',
                'Irregular borders', 
                'Multiple colors or color changes',
                'Diameter larger than 6mm',
                'Evolving size, shape, or color'
            ],
            'risk_factors': ['UV exposure', 'Fair skin', 'Family history', 'Many moles']
        },
        {
            'code': 'NV',
            'name': 'Melanocytic Nevus',
            'description': 'Common benign skin lesions, also known as moles, formed by clusters of melanocytes.',
            'image_filename': 'nevus_example.jpg',
            'characteristics': [
                'Usually symmetrical',
                'Smooth, regular borders',
                'Uniform color (brown, black, or flesh-colored)',
                'Stable size and appearance',
                'Can be flat or raised'
            ],
            'risk_factors': ['Genetics', 'Sun exposure', 'Hormonal changes']
        },
        {
            'code': 'BCC',
            'name': 'Basal Cell Carcinoma',
            'description': 'The most common type of skin cancer, arising from basal cells in the lower part of the epidermis.',
            'image_filename': 'bcc_example.jpg',
            'characteristics': [
                'Pearl-like or waxy appearance',
                'Raised edges with central depression',
                'May bleed easily',
                'Slow-growing',
                'Often on sun-exposed areas'
            ],
            'risk_factors': ['Chronic sun exposure', 'Fair skin', 'Age', 'Male gender']
        },
        {
            'code': 'AKIEC',
            'name': 'Actinic Keratosis',
            'description': 'Precancerous lesions caused by sun damage that may develop into squamous cell carcinoma.',
            'image_filename': 'ak_example.jpg',
            'characteristics': [
                'Rough, scaly texture',
                'Red, brown, or skin-colored',
                'Flat or slightly raised',
                'May be tender or itchy',
                'On sun-exposed areas'
            ],
            'risk_factors': ['Chronic sun exposure', 'Fair skin', 'Age over 40', 'Immunosuppression']
        },
        {
            'code': 'BKL',
            'name': 'Benign Keratosis',
            'description': 'Non-cancerous skin growths including seborrheic keratoses that appear with aging.',
            'image_filename': 'bkl_example.jpg',
            'characteristics': [
                'Waxy, "stuck-on" appearance',
                'Brown, black, or tan color',
                'Well-defined borders',
                'Rough or smooth surface',
                'Various sizes'
            ],
            'risk_factors': ['Aging', 'Genetics', 'Sun exposure history']
        },
        {
            'code': 'DF',
            'name': 'Dermatofibroma',
            'description': 'Benign skin nodules composed of fibrous tissue, often appearing after minor skin trauma.',
            'image_filename': 'df_example.jpg',
            'characteristics': [
                'Firm, small nodules',
                'Brown, red, or pink color',
                'Dimples when pinched',
                'Usually on legs or arms',
                'Slow-growing'
            ],
            'risk_factors': ['Minor skin trauma', 'Insect bites', 'More common in women']
        },
        {
            'code': 'VASC',
            'name': 'Vascular Lesions',
            'description': 'Lesions involving blood vessels, including hemangiomas, cherry angiomas, and other vascular malformations.',
            'image_filename': 'vasc_example.jpg',
            'characteristics': [
                'Red, purple, or blue color',
                'May blanch with pressure',
                'Smooth or raised surface',
                'Various sizes',
                'Well-defined borders'
            ],
            'risk_factors': ['Age', 'Genetics', 'Hormonal changes', 'Sun exposure']
        },
        {
            'code': 'SCC',
            'name': 'Squamous Cell Carcinoma',
            'description': 'Second most common skin cancer, arising from squamous cells in the upper layers of skin.',
            'image_filename': 'scc_example.jpg',
            'characteristics': [
                'Scaly, rough surface',
                'Red, inflamed appearance',
                'May ulcerate or crust',
                'Rapid growth',
                'On sun-exposed areas'
            ],
            'risk_factors': ['Chronic sun exposure', 'Fair skin', 'Immunosuppression', 'HPV infection']
        }
    ]
    
    context = {
        'lesion_data': lesion_data
    }
    return render(request, 'lesion_analyzer/lesion_types.html', context)

def how_it_works(request):
    """Page explaining the AI analysis procedure"""
    return render(request, 'lesion_analyzer/how_it_works.html')

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
