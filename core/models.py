# core/models.py
from django.db import models
from datetime import datetime

class RedditPost(models.Model):
    subreddit = models.CharField(max_length=255)
    title = models.TextField()
    score = models.IntegerField()
    comments = models.IntegerField()
    post_time = models.DateTimeField()
    post_content = models.TextField(blank=True, null=True) # 允許空值
    url = models.URLField(max_length=500) # 設置一個更大的最大長度以避免截斷長URL

    temp_field = models.BooleanField(default=False, blank=True, null=True)

    def __str__(self):
        return self.title

    class Meta:
        ordering = ['-post_time'] # 依發文時間排序
        # --- 請確保這一行沒有被註釋掉！ ---
        unique_together = (('subreddit', 'title', 'post_time'),) # 根據這三個欄位判斷唯一性
        # --- 請確保這一行沒有被註釋掉！ ---