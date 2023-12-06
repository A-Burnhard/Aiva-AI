from django.urls import path
from  . import views
from .views import HomeView

urlpatterns = [ 
    path('', HomeView.as_view(), name='home'),
]