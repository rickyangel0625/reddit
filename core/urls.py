# core/urls.py
from django.urls import path
from . import views
from django.views.generic import RedirectView # 用於重定向，可選

urlpatterns = [
    # path('', views.post_list, name='post_list'), # 將根路徑映射到 post_list 視圖
    path('', views.post_list_view, name='post_list'),
    path('lda_result/', views.lda_visualization_view, name='lda_result'),
]