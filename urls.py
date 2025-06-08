from django.urls import path
from . import views

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('analyze', views.analyze_audio, name='analyze_audio'),
]
