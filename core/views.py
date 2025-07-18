# core/views.py
from django.shortcuts import render
from .models import RedditPost # 從同一個應用程式導入 RedditPost 模型
import os
import base64
from django.conf import settings # 引入 settings
from django.db.models import Count # 用於動態獲取子版塊

def post_list(request):
    # 從資料庫獲取所有 RedditPost 對象
    # 這裡我們按照發文時間降序排列 (最新在前)，這與您模型中的 Meta.ordering 一致
    posts = RedditPost.objects.all()
    
    # 將數據傳遞給模板
    context = {
        'posts': posts
    }
    return render(request, 'core/post_list.html', context)


def post_list_view(request):
    # --- 1. 獲取所有貼文 (取消篩選功能) ---
    posts = RedditPost.objects.all() 

    # --- 2. 準備詞雲圖片 Base64 字串 
    wordcloud_b64 = None
    # 組合詞雲圖片的完整檔案路徑
    # settings.BASE_DIR 是您的專案根目錄
    wordcloud_path = os.path.join(settings.BASE_DIR, 'core', 'static', 'core', 'images', 'wordcloud.png') # 這裡改為 .png
    try:
        with open(wordcloud_path, "rb") as image_file:
            wordcloud_b64 = base64.b64encode(image_file.read()).decode('utf-8')
            wordcloud_b64 = f"data:image/png;base64,{wordcloud_b64}" # 這裡改為 data:image/png
    except FileNotFoundError:
        print(f"錯誤：找不到詞雲圖片檔案：{wordcloud_path}")
    except Exception as e:
        print(f"詞雲圖片轉換錯誤：{e}")

    # --- 3. LDA 可視化不在此處處理，透過獨立 URL 連結 ---
    context = {
        'posts': posts, # <-- NameError 發生在這裡，因為上面的 posts = ... 沒有被執行或定義
        'wordcloud_b64': wordcloud_b64,
    }
    # 渲染到修改後的 post_list.html
    return render(request, 'core/post_list.html', context)
    


def lda_visualization_view(request):
    context = {} 
    return render(request, 'core/lda_visualization.html', context)