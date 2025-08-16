import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from datetime import datetime
import gensim
from gensim import corpora
from nltk.corpus import stopwords
import nltk
from wordcloud import WordCloud
import json
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import seaborn as sns
import numpy as np
from scipy.stats import shapiro, skew, kurtosis # 從 scipy.stats 導入 shapiro, skew, kurtosis
import statsmodels.api as sm

# 確保必要的 NLTK 資源已下載
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- 設置 Matplotlib 支持中文字符 ---
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
# --- 設置結束 ---

# ========== 1. 讀取 JSON 數據 ==========
print("=== 1. 讀取 JSON 數據 ===")
with open('reddit_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# 確保日期格式正確
df['Post Time'] = pd.to_datetime(df['Post Time'], errors='coerce')
print(f"已讀取 {len(df)} 筆數據。")

# 移除內容為空的貼文，以免影響分析
df.dropna(subset=['Post Content'], inplace=True)
df = df[df['Post Content'].str.strip() != '']
print(f"移除空內容貼文後，剩餘 {len(df)} 筆數據。")

# ========== 2. 關鍵詞分析（Unigram & Bigram） ==========
print("\n=== 2. 關鍵詞提取分析 ===")

# 進行 unigram 分析
unigram_vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words=list(stopwords.words('english')), max_features=20)
X_uni = unigram_vectorizer.fit_transform(df['Post Content'].fillna(''))
unigrams = unigram_vectorizer.get_feature_names_out()
print("最常見的單詞 (Unigrams):", unigrams)

# 進行 bigram 分析
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words=list(stopwords.words('english')), max_features=20)
X_bi = bigram_vectorizer.fit_transform(df['Post Content'].fillna(''))
bigrams = bigram_vectorizer.get_feature_names_out()
print("最常見的二元詞組 (Bigrams):", bigrams)

# 創建詞雲
all_words = ' '.join([text for text in df['Post Content'].fillna('')])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Post Content')
plt.savefig('wordcloud.png')
plt.show()
plt.close()


# ========== 3. 時間序列分析 ==========
print("\n=== 3. 時間序列分析 ===")
df_time_series = df.set_index('Post Time')
daily_posts = df_time_series.resample('D').size().fillna(0)

plt.figure(figsize=(15, 7))
plt.plot(daily_posts.index, daily_posts.values, marker='o', markersize=3, linestyle='-', color='skyblue')
plt.title('Daily Post Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Posts')
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('daily_post_volume.png')
plt.show()
plt.close()

print("每日貼文數量統計：")
print(daily_posts.describe())


# ========== 4. 情感分析 ==========
print("\n=== 4. 情感分析 ===")
df['Sentiment Score'] = df['Post Content'].fillna('').apply(lambda text: TextBlob(text).sentiment.polarity)
print("情感分數統計：")
print(df['Sentiment Score'].describe())


# ========== 5. 評論活躍度分析 ==========
print("\n=== 5. 評論活躍度分析 ===")
df['Comments'] = pd.to_numeric(df['Comments'], errors='coerce').fillna(0)
print("評論數統計：")
print(df['Comments'].describe())


# ========== 新增：數據分佈特性檢定 (常態性、偏度、峰度) ==========
print("\n=== 數據分佈特性檢定 ===")

# 情感分數 (Sentiment Score) 的檢定
sentiment_data = df['Sentiment Score'].dropna()
if len(sentiment_data) > 1: # 確保有足夠的數據進行檢定
    sentiment_stat, sentiment_p = shapiro(sentiment_data)
    sentiment_skewness = skew(sentiment_data)
    sentiment_kurtosis = kurtosis(sentiment_data, fisher=False) # fisher=False 得到 Pearson's kurtosis (常態為 3)
    
    print(f"情感分數 (Sentiment Score) 檢定結果:")
    print(f"  Shapiro-Wilk Test Statistic: {sentiment_stat:.4f}, P-value: {sentiment_p:.4f}")
    print(f"  Skewness (偏度): {sentiment_skewness:.4f} (常態分佈應接近 0)")
    print(f"  Kurtosis (峰度): {sentiment_kurtosis:.4f} (常態分佈應接近 3)")

    if sentiment_p > 0.05 and abs(sentiment_skewness) < 0.5 and abs(sentiment_kurtosis - 3) < 1: # 寬鬆判斷，可調整閾值
        print("  => 情感分數分佈 **可能接近常態分佈**，適用三倍標準差原則。")
    else:
        print("  => 情感分數分佈 **不完全符合常態分佈**。雖然 P 值可能很小，但偏度和峰度可作為參考。")
else:
    print("  情感分數數據量不足，無法進行常態性檢定。")

print("-" * 50)

# 評論數 (Comments Count) 的檢定
comments_data = df['Comments'].dropna()
if len(comments_data) > 1: # 確保有足夠的數據進行檢定
    comments_stat, comments_p = shapiro(comments_data)
    comments_skewness = skew(comments_data)
    comments_kurtosis = kurtosis(comments_data, fisher=False)
    
    print(f"評論數 (Comments Count) 檢定結果:")
    print(f"  Shapiro-Wilk Test Statistic: {comments_stat:.4f}, P-value: {comments_p:.4f}")
    print(f"  Skewness (偏度): {comments_skewness:.4f} (常態分佈應接近 0)")
    print(f"  Kurtosis (峰度): {comments_kurtosis:.4f} (常態分佈應接近 3)")

    if comments_p > 0.05 and abs(comments_skewness) < 0.5 and abs(comments_kurtosis - 3) < 1:
        print("  => 評論數分佈 **可能接近常態分佈**，適用三倍標準差原則。")
    else:
        print("  => 評論數分佈 **不完全符合常態分佈**。P 值很小通常意味著顯著偏離常態。")
else:
    print("  評論數數據量不足，無法進行常態性檢定。")


# ========== 6. 主題建模 (LDA) ==========
print("\n=== 6. 主題建模 (LDA) ===")
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = gensim.utils.simple_preprocess(str(text), deacc=True)
    return [token for token in tokens if token not in stop_words and len(token) > 2]

df['Tokens'] = df['Post Content'].fillna('').apply(preprocess)

dictionary = corpora.Dictionary(df['Tokens'])
corpus = [dictionary.doc2bow(text) for text in df['Tokens']]

num_topics = 5
lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=100)

print("LDA Topics:")
for idx, topic in lda_model.print_topics():
    print(f"Topic {idx+1}: {topic}")

def get_lda_topic(doc_bow):
    if not doc_bow:
        return None
    topic_distribution = lda_model[doc_bow]
    if not topic_distribution:
        return None
    return max(topic_distribution, key=lambda x: x[1])[0] + 1

df['LDA_Topic'] = [get_lda_topic(doc_bow) for doc_bow in corpus]
print("\n部分貼文及其分配的主題：")
print(df[['Title', 'Post Content', 'LDA_Topic']].head())

vis = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)
pyLDAvis.save_html(vis, 'lda_visualization.html')
print("LDA 可視化結果已保存至 lda_visualization.html")


# ========== 7. 詐騙行為揭示 (Scam Revelation) ==========
print("\n=== 7. 詐騙行為揭示 (Scam Revelation) ===")

df_reveal = df.copy()

mean_sentiment = df_reveal['Sentiment Score'].mean()
std_sentiment = df_reveal['Sentiment Score'].std()
mean_comments = df_reveal['Comments'].mean()
std_comments = df_reveal['Comments'].std()

print(f"\n--- 揭示詐騙的統計閾值 ---")
print(f"情感分數平均值: {mean_sentiment:.4f}, 標準差: {std_sentiment:.4f}")
print(f"評論數平均值: {mean_comments:.2f}, 標準差: {std_comments:.2f}")

lower_sentiment_threshold = mean_sentiment - (3 * std_sentiment)
upper_sentiment_threshold = mean_sentiment + (3 * std_sentiment)
upper_comments_threshold = mean_comments + (3 * std_comments)

print(f"情感異常閾值 (Sentiment Score < {lower_sentiment_threshold:.4f} 或 > {upper_sentiment_threshold:.4f})")
print(f"互動熱度異常閾值 (Comments > {upper_comments_threshold:.2f})")

df_reveal['is_extreme_sentiment'] = (df_reveal['Sentiment Score'] < lower_sentiment_threshold) | \
                                     (df_reveal['Sentiment Score'] > upper_sentiment_threshold)
df_reveal['is_high_comments'] = (df_reveal['Comments'] > upper_comments_threshold)
df_reveal['is_scam_revealing_post'] = df_reveal['is_extreme_sentiment'] | df_reveal['is_high_comments']

revealed_scam_posts = df_reveal[df_reveal['is_scam_revealing_post']]

print(f"\n--- 詐騙行為揭示結果總覽 ---")
print(f"總貼文數: {len(df_reveal)}")
print(f"被「情感異常指標」揭示的貼文數: {df_reveal['is_extreme_sentiment'].sum()}")
print(f"被「互動熱度異常指標」揭示的貼文數: {df_reveal['is_high_comments'].sum()}")
print(f"總計被「揭示詐騙行為」的貼文數: {len(revealed_scam_posts)}")
print(f"被揭示的貼文佔比: {(len(revealed_scam_posts) / len(df_reveal) * 100):.2f}%")

print(f"\n--- 被揭示詐騙貼文的內容與時間趨勢分析 ---")

print("\n部分被揭示的詐騙相關貼文範例 (前5篇):")
if not revealed_scam_posts.empty:
    print(revealed_scam_posts[['Post Time', 'Subreddit', 'Title', 'Sentiment Score', 'Comments']].head())
else:
    print("沒有符合條件的被揭示貼文。")

if not revealed_scam_posts.empty:
    revealed_scam_posts_copy = revealed_scam_posts.copy()
    revealed_scam_posts_copy['Post Month'] = revealed_scam_posts_copy['Post Time'].dt.to_period('M')
    monthly_revealed_counts = revealed_scam_posts_copy['Post Month'].value_counts().sort_index()

    print("\n被揭示詐騙貼文的每月分佈:")
    print(monthly_revealed_counts)

    plt.figure(figsize=(12, 6))
    monthly_revealed_counts.plot(kind='bar', color='skyblue')
    plt.title('每月被揭示詐騙貼文數量分佈 (Monthly Distribution of Revealed Scam Posts)')
    plt.xlabel('月份')
    plt.ylabel('被揭示貼文數量')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig('revealed_scam_posts_monthly_distribution.png')
    plt.show()
    plt.close()
else:
    print("\n無法生成被揭示詐騙貼文的每月分佈圖，因為沒有被揭示的貼文。")

if 'LDA_Topic' in df_reveal.columns and not revealed_scam_posts.empty:
    print("\n被揭示詐騙貼文的主題分佈:")
    revealed_scam_posts_copy = revealed_scam_posts.copy()
    revealed_scam_posts_copy['LDA_Topic'] = pd.to_numeric(revealed_scam_posts_copy['LDA_Topic'], errors='coerce').fillna(-1)
    
    valid_revealed_topics = revealed_scam_posts_copy[revealed_scam_posts_copy['LDA_Topic'] != -1]['LDA_Topic'].value_counts(normalize=True)
    print(valid_revealed_topics)

    if not valid_revealed_topics.empty:
        print(f"--- 準備繪製圓餅圖：主題分佈數據如下 ---")
        print(valid_revealed_topics)

        topic_numbers = valid_revealed_topics.index
        proportions = valid_revealed_topics.values

        legend_labels = [f"主題 {int(num)} ({prop*100:.1f}%)" for num, prop in zip(topic_numbers, proportions)]

        plt.figure(figsize=(10, 10))
        colors = plt.cm.viridis(np.linspace(0, 1, len(proportions)))
        wedges, texts, autotexts = plt.pie(proportions, autopct='%1.1f%%', startangle=90, colors=colors, pctdistance=0.85)
        
        plt.title('被揭示詐騙貼文的主題分佈 (LDA Topic Distribution of Revealed Scam Posts)', fontsize=14)
        plt.ylabel('')

        plt.legend(wedges, legend_labels, title="LDA 主題", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
        
        plt.tight_layout()
        plt.savefig('revealed_scam_topics_pie.png')
        plt.show()
        plt.close()
    else:
        print("--- 警告：沒有有效的被揭示貼文主題可供繪製圓餅圖。 ---")
        print(f"有效被揭示主題數據為空: {valid_revealed_topics.empty}")
elif 'LDA_Topic' not in df_reveal.columns:
    print("\n'LDA_Topic' 欄位未找到，無法分析被揭示貼文的主題分佈。")
else:
    print("\n沒有被揭示的貼文，無法分析主題分佈。")


# ========== 新增視覺化: 情感分數分佈圖標示閾值 ==========
plt.figure(figsize=(10, 6))
sns.histplot(df_reveal['Sentiment Score'], bins=30, kde=True, color='skyblue', label='所有貼文情感分數')
plt.axvline(lower_sentiment_threshold, color='red', linestyle='--', label='下閾值 (-3 SD)')
plt.axvline(upper_sentiment_threshold, color='red', linestyle='--', label='上閾值 (+3 SD)')
plt.title('情感分數分佈與揭示閾值 (Sentiment Score Distribution with Revelation Thresholds)')
plt.xlabel('情感分數')
plt.ylabel('貼文數量')
plt.legend()
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('sentiment_distribution_with_thresholds.png')
plt.show()
plt.close()


# ========== 新增視覺化: 評論數分佈圖標示閾值 ==========
plt.figure(figsize=(10, 6))
sns.histplot(df_reveal['Comments'], bins=30, kde=True, color='lightgreen', label='所有貼文評論數')
plt.axvline(upper_comments_threshold, color='red', linestyle='--', label='上閾值 (+3 SD)')
plt.title('評論數量分佈與揭示閾值 (Comment Count Distribution with Revelation Thresholds)')
plt.xlabel('評論數量')
plt.ylabel('貼文數量')
plt.legend()
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('comment_distribution_with_thresholds.png')
plt.show()
plt.close()

# 情感分數 Q-Q 圖
sm.qqplot(df['Sentiment Score'].dropna(), line='s') # line='s'表示標準常態分佈的理論線
plt.title('Q-Q Plot for Sentiment Score')
plt.xlabel('Theoretical Quantiles (Normal Distribution)')
plt.ylabel('Sample Quantiles (Sentiment Score)')
plt.savefig('qq_plot_sentiment.png')
plt.show()
plt.close()

# 評論數 Q-Q 圖
sm.qqplot(df['Comments'].dropna(), line='s')
plt.title('Q-Q Plot for Comments Count')
plt.xlabel('Theoretical Quantiles (Normal Distribution)')
plt.ylabel('Sample Quantiles (Comments Count)')
plt.savefig('qq_plot_comments.png')
plt.show()
plt.close()

print("\n數據分析和揭示過程完成！所有圖表已生成。")