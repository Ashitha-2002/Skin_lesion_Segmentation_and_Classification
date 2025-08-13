from django.contrib import admin
from .models import LesionAnalysis

@admin.register(LesionAnalysis)
class LesionAnalysisAdmin(admin.ModelAdmin):
    list_display = ['id', 'predicted_class', 'confidence_score', 'created_at']
    list_filter = ['predicted_class', 'created_at']
    readonly_fields = ['created_at']
    search_fields = ['predicted_class']