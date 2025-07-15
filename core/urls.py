# core/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.post_list, name='post_list'), # 將根路徑映射到 post_list 視圖
]