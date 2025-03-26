# VesterAI: Financial Sentiment Intelligence Chatbot

**Author:** Tandel Raj  
**University Capstone Project**  
**Model Deployment: Fine-tuned LLaMA 2 | Frontend: Streamlit Dashboard & Chatbot**

---

## 📌 Overview
VesterAI is an AI-powered financial sentiment analysis and market forecasting system built as a solo capstone project. It integrates natural language processing, machine learning, and LLM-based summarization to help investors make data-driven decisions.

The system collects real-time sentiment from Twitter, financial news, and optionally Reddit, merges it with technical indicators (RSI, MACD, OBV, lag returns), and uses fine-tuned LLaMA 2 to generate natural-language insights for stock prediction. The system includes:

- Sentiment scraping + analysis
- Feature engineering from stock prices
- Predictive modeling (classification & regression)
- Instruction fine-tuning of LLaMA 2 for market summaries
- A chatbot interface and visual dashboard

---

## 📁 Folder Structure
```
VesterAI/
├── 01_data_scraping_twitter.ipynb         # Twitter scraping using snscrape
├── 02_sentiment_labeling.ipynb            # Applies FinBERT, RoBERTa, VADER
├── 03_news_scraping_and_sentiment.ipynb   # Financial news + sentiment
├── 04_data_merging_and_feature_engineering.ipynb
├── 05_model_training_prediction.ipynb     # ML model training + evaluation
├── 06_llm_insights_generation.ipynb       # Generates prompts for LLM tuning
├── 07_llama2_finetuning.ipynb             # Fine-tunes LLaMA 2 (7B) using LoRA
├── 08_chatbot_interface.ipynb             # Streamlit app with LLaMA 2 chatbot
├── 09_dashboard_visualization.ipynb       # Technical charts + sentiment trends
├── streamlit_app/
│   └── app.py                              # App launcher for HPC deployment
├── models/                                 # Saved fine-tuned LLaMA 2 weights
├── data/
│   ├── raw/                                # Tweets, news, price data (CSV)
│   ├── processed/                          # Sentiment-labeled datasets
│   └── final/                              # Final merged model-ready dataset
├── requirements.txt                        # All Python dependencies
└── README.md                               # Project instructions (this file)
```

---

## ⚙️ Features
- **Multi-source sentiment** (Twitter, news, Reddit)
- **Triple sentiment modeling** (FinBERT, RoBERTa, VADER)
- **Feature-rich forecasting models** (RSI, MACD, OBV, returns)
- **Fine-tuned LLaMA 2 (7B)** for natural-language summaries
- **Interactive chatbot** using Streamlit
- **Analytics dashboard** for EDA and visual storytelling

---

## 🧠 Technologies Used
- Python 3.8+
- Transformers (Hugging Face)
- PEFT / LoRA (for LLaMA 2 fine-tuning)
- Scikit-learn, XGBoost, TA-lib
- Pandas, NumPy, Matplotlib, Plotly
- Streamlit
- snscrape (Twitter scraping)

---

## 📈 Model Summary
| Task              | Model              | Metric        |
|------------------|-------------------|---------------|
| Classification   | Random Forest      | Accuracy: ~67%|
| Regression       | XGBoost            | R²: ~0.30     |
| LLM Generation   | LLaMA 2 (7B, LoRA) | Summary Accuracy via Prompt Testing |

---

## 🚀 Running the Project
1. Clone the repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run notebooks in order: 01 to 09
4. To launch chatbot:
   ```bash
   cd streamlit_app
   streamlit run app.py
   ```

---

## 📺 Demo Videos

Click below to watch the demos:

## VesterAI Dashboard & Architecture

[![VesterAI Chatbot Demo](https://img.youtube.com/vi/cJuhyeQBRO0/0.jpg)](https://www.youtube.com/watch?v=cJuhyeQBRO0)

## VesterAI Chatbot Demo

[![VesterAI Dashboard & Architecture](https://img.youtube.com/vi/QSBXRswd3lY/0.jpg)](https://www.youtube.com/watch?v=QSBXRswd3lY)

---

## 📚 References
- Moradi-Kamali et al. (2025). *Revisiting Financial Sentiment Analysis with LLMs*. arXiv.
- Araci, D. (2019). *FinBERT*. arXiv.
- Yang et al. (2023). *FinGPT*. arXiv.
- Asgarov, A. (2023). *Market Prediction Using Sentiment*. arXiv.
- Mamillapalli et al. (2024). *GRUvader*. Mathematics.

*Full reference list is available in the final report.*

---

## 👨‍💻 Author
**Tandel Raj**  
Solo developer and researcher  
Please cite or credit this repository if you build upon this work.

---

## 📬 Contact
For questions or collaboration: tandel.r@northeastern.edu

---

Thank you for checking out VesterAI!

