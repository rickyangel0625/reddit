# core/views.py
from django.shortcuts import render
from .models import RedditPost # 從同一個應用程式導入 RedditPost 模型

def post_list(request):
    # 從資料庫獲取所有 RedditPost 對象
    # 這裡我們按照發文時間降序排列 (最新在前)，這與您模型中的 Meta.ordering 一致
    posts = RedditPost.objects.all()
    
    # 將數據傳遞給模板
    context = {
        'posts': posts
    }
    return render(request, 'core/post_list.html', context)