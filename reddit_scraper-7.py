import praw
import prawcore
import json
import time
from datetime import datetime
from pytz import utc # Add this line at the top with other imports
import os
from dotenv import load_dotenv

# 載入 .env 檔案中的環境變數
load_dotenv()

# 初始化 Reddit API
reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent=os.getenv('REDDIT_USER_AGENT')
)

# 設定要抓取的子版塊
subreddits = ['Fraudsters', 'Scams', 'Romancescam', 'ScamBait', 'LegalAdvice']
# subreddits = ['Frauditors', 'Scams', 'Romancescam', 'ScamBait', 'LegalAdvice']
# subreddits = ['Fraudsters', 'Scams', 'Assistance', 'ScamBait', 'LegalAdvice', 'personalfinance', 'Frauditors', 'CryptoCurrency']

# 創建數據存儲結構
data = []

def fetch_posts(subreddit_name, limit=60):
    """抓取指定子版塊的熱門貼文"""
    post_count = 0 # Initialize counter for this subreddit
    try:
        subreddit = reddit.subreddit(subreddit_name)
        print(f"Fetching data from subreddit: {subreddit_name}")
        
        for post in subreddit.hot(limit=limit):  # 每個子版塊抓取 `limit` 篇熱門貼文
            post_time = datetime.fromtimestamp(post.created_utc, utc).strftime('%Y-%m-%d %H:%M:%S')
            data.append({
                "Subreddit": subreddit_name,
                "Title": post.title,
                "Score": post.score,
                "Comments": post.num_comments,
                "Post Time": post_time,
                "Post Content": post.selftext[:1000],  # 限制內容長度
                "URL": post.url
            })
            post_count += 1 # Increment counter for each post added
        print(f"Successfully fetched {post_count} posts from r/{subreddit_name}.") # Print count here
        return post_count # Return the count
    
    except prawcore.exceptions.NotFound:
        print(f"Subreddit {subreddit_name} not found or is private.")
    except prawcore.exceptions.Forbidden:
        print(f"Access to subreddit {subreddit_name} is forbidden.")
    except prawcore.exceptions.ResponseException as e:
        print(f"Response error when fetching {subreddit_name}: {e}")
    except Exception as e:
        print(f"Unexpected error fetching {subreddit_name}: {e}")

        

# 主程式邏輯
total_posts_grabbed = 0
for subreddit_name in subreddits:
    count = fetch_posts(subreddit_name, limit=60)  # 降低抓取數量以提高穩定性
    total_posts_grabbed += count
    time.sleep(5)  # 每次請求後等待 5 秒，避免觸發 API 限制

# 將數據儲存為 JSON 檔案
output_file = 'reddit_data.json'
try:
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"數據抓取完成，已儲存到 {output_file}。")
    print(f"Total posts grabbed across all subreddits: {total_posts_grabbed}") # Print total count
except Exception as e:
    print(f"Error saving data to {output_file}: {e}")

# hot：熱門帖子，受 Reddit 熱度算法影響。
# new：最新的帖子。
# top：根據時間範圍排序的熱門帖子。
# rising：正在快速上升的帖子。
# controversial：最具爭議性的帖子。
# comments：子版塊中的評論（不是貼文）。