# Reddit 詐騙貼文平台
這是一個基於 Django 的 Web 應用程式，旨在透過 Reddit API 獲取相關貼文，並對詐騙議題進行深度分析，提供關鍵數據洞察。

[佈署網站](https://reddit-cwzn.onrender.com)

![首頁截圖](screenshots/homepage.png)
![LDA 可視化截圖](screenshots/lda_visualization.png)

## 功能概覽 (Features)

* **Reddit 貼文列表：** 呈現從 Reddit API 獲取的詐騙相關貼文。
* **詞雲分析：** 將所有貼文內容的熱門關鍵字視覺化，直觀呈現詞語頻率與重要性。
* **LDA 主題模型可視化：** 文字探勘運用LDA(Latent Dirichlet Allocation)主題模型，揭示貼文中的潛在主題結構及其分佈，提供數據驅動的洞察。
* **Django 框架構建：** 使用 Django 框架開發。
* **Render 部署：** 部署至 Render 雲端平台。

## 技術棧 (Tech Stack)

* **後端:** Python, Django, Gunicorn, WhiteNoise
* **資料庫:** PostgreSQL/SQLite 
* **資料分析/處理:** NLTK, Gensim, WordCloud, Matplotlib
* **部署:** Render
* **版本控制:** Git
* **前端:** HTML, CSS

## 未來改進 (Future Improvements)
* 增加用戶互動功能，允許用戶根據子版塊 (subreddits) 進行貼文篩選，以提供更精準的數據檢視。
