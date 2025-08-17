from django.urls import path
from . import views
from django.shortcuts import render

app_name = 'lesion_analyzer'

urlpatterns = [
     path('', views.home, name='home'),
    path('upload/', views.upload_image, name='upload'),
    path('results/<int:analysis_id>/', views.view_results, name='results'),
    path('history/', views.analysis_history, name='history'),
    path('lesion-types/', views.lesion_types, name='lesion_types'),
    path('how-it-works/', views.how_it_works, name='how_it_works'),
]