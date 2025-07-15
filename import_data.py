# D:\codes\reddit\import_data.py

import os
import django
import json
from datetime import datetime
from pytz import utc # Ensure pytz is installed and imported

# 設置 Django 環境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'reddit.settings') # 替換 'reddit' 為您的主專案名稱
django.setup()

# 導入您的模型
from core.models import RedditPost # 確保 'core' 是您應用程式的名稱

# 數據文件路徑
# 確保 reddit_data.json 檔案在與此腳本相同的目錄下
current_dir = os.path.dirname(__file__)
data_file_path = os.path.join(current_dir, 'reddit_data.json')

print(f"嘗試從 '{data_file_path}' 讀取數據...")

try:
    with open(data_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"成功讀取 {len(data)} 條數據。")
except FileNotFoundError:
    print(f"錯誤：找不到文件 '{data_file_path}'。請確保文件存在並在正確的路徑。")
    exit()
except json.JSONDecodeError as e:
    print(f"錯誤：解析 JSON 文件失敗：{e}")
    exit()

# 遍歷數據並保存到資料庫
posts_created = 0
posts_skipped = 0

for item in data:
    # 檢查是否已存在相同標題和內容的貼文，避免重複導入
    # 考慮 Reddit 貼文是否有唯一 ID，使用 ID 會更準確
    # 這裡暫時使用標題和發文時間作為唯一判斷依據（需要確保 Post Time 精確到秒）
    
    # 將時間字符串轉換為 datetime 對象，並設置時區資訊
    try:
        # 由於 Reddit API 返回的 'created_utc' 是 Unix 時間戳，
        # 如果您的 JSON 是直接從 PRAW 的 created_utc 轉換而來，
        # 那麼這裡應該處理為 datetime.fromtimestamp(item['created_utc'], tz=utc)
        # 如果您的 JSON 已經是格式化字符串，例如 '2023-10-26 12:34:56'，則需要用 strptime
        
        # 根據您 reddit_scraper-6.py 的輸出格式 ('%Y-%m-%d %H:%M:%S')
        post_time_str = item.get('Post Time')
        if post_time_str:
            post_time_dt = datetime.strptime(post_time_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=utc)
        else:
            print(f"警告：跳過缺少 'Post Time' 的貼文：{item.get('Title', '未知標題')}")
            posts_skipped += 1
            continue

        existing_post = RedditPost.objects.filter(
            title=item.get('Title'),
            subreddit=item.get('Subreddit'),
            post_time=post_time_dt # 用精確的時間戳來判斷重複
        ).first()

        if not existing_post:
            RedditPost.objects.create(
                subreddit=item.get('Subreddit'),
                title=item.get('Title'),
                score=item.get('Score'),
                comments=item.get('Comments'),
                post_time=post_time_dt,
                post_content=item.get('Post Content'),
                url=item.get('URL')
            )
            posts_created += 1
        else:
            posts_skipped += 1
            # print(f"已存在，跳過：{item.get('Title')}") # 如果想看被跳過的貼文，取消註釋
    except Exception as e:
        print(f"導入數據時發生錯誤：{e}，數據：{item.get('Title', 'N/A')}")
        posts_skipped += 1

print(f"數據導入完成。成功創建 {posts_created} 條貼文，跳過 {posts_skipped} 條貼文（可能已存在或數據有問題）。")