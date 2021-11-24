from django.urls import path
from django.views.generic.base import TemplateView
from house_project.views import HousePrediction


urlpatterns = [
    path("", TemplateView.as_view(template_name="index.html"), name="home"),
    path("house_prediction", HousePrediction.as_view(), name="house_prediction"),
]
