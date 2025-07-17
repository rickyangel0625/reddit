"""
URL configuration for reddit project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# reddit/urls.py 
from django.contrib import admin
from django.urls import path, include # 導入 include
from django.conf import settings # 引入 settings
from django.conf.urls.static import static # 引入 static 函數

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('core.urls')), # 將根路徑 ('') 導向 core 應用程式的 urls.py
]

# 在開發環境下服務靜態檔案 (Render 會自動處理)
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
