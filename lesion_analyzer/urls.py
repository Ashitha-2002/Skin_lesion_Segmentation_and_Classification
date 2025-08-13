from django.urls import path
from . import views

app_name = 'lesion_analyzer'

urlpatterns = [
    path('', views.upload_image, name='upload'),
    path('results/<int:analysis_id>/', views.view_results, name='results'),
    path('history/', views.analysis_history, name='history'),
]