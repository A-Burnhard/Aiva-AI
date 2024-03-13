from django.contrib import admin
from django.urls import path

from api import views
from api.views import *

urlpatterns = [

    path('', views.ProcessorView.as_view(), name ="main"),
    path('chat/', views.ChatView.as_view(), name ="chat"),
    path('test/', views.TestView.as_view(), name ="test"),

]