import streamlit as st
import pandas as pd
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load fine-tuned model
model_path = "../models/llama2_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load data
data_path = "../data/processed/AAPL_model_data.csv"
df = pd.read_csv(data_path, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Prompt builder
def build_prompt(user_question, row):
    context = f"""Date: {row['Date'].strftime('%Y-%m-%d')}
Stock Close: {row['Close']:.2f}
Return: {row['return']:.4f}
Twitter Sentiment: {row['twitter_sentiment']:.2f}
News Sentiment: {row['news_sentiment']:.2f}
Reddit Sentiment: {row.get('reddit_sentiment', 0):.2f}
RSI: {row['rsi_14']:.2f}
MACD: {row['macd']:.4f}
OBV: {row['obv']:.2f}"""

    return f"""### Instruction:
{user_question}

### Input:
{context}

### Response:
"""

# LLM response generator
def get_llm_response(question, row):
    prompt = build_prompt(question, row)
    output = llm(prompt, max_new_tokens=200)[0]["generated_text"]
    return output.replace(prompt, "").strip()

# Streamlit UI
st.set_page_config(page_title="VesterAI Chatbot", layout="wide")
st.title("VesterAI - Financial Sentiment Chatbot")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load prediction file if exists
prediction_file = "../data/processed/AAPL_predictions.csv"
if os.path.exists(prediction_file):
    prediction_df = pd.read_csv(prediction_file, parse_dates=["Date"])
else:
    prediction_df = pd.DataFrame()

# Date selector
available_dates = df["Date"].dt.strftime('%Y-%m-%d').tolist()
selected_date_str = st.selectbox("Select a date to analyze:", options=available_dates[::-1])
selected_date = pd.to_datetime(selected_date_str)
row = df[df["Date"] == selected_date].iloc[0]

# User input
user_input = st.text_input("Ask a question about the selected date's market sentiment or outlook:")

# Generate response
if st.button("Generate Insight"):
    if user_input:
        response = get_llm_response(user_input, row)

        # Save to chat history
        st.session_state.chat_history.append({
            "date": selected_date_str,
            "question": user_input,
            "response": response
        })

        st.write("**Response:**")
        st.markdown(response)
    else:
        st.warning("Please enter a question.")

# Show chat history
if st.session_state.chat_history:
    st.subheader("Chat History")
    for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
        st.markdown(f"**{i}. Date: {chat['date']}**")
        st.markdown(f"- **Q:** {chat['question']}")
        st.markdown(f"- **A:** {chat['response']}")
        st.markdown("---")

# Export chat history to CSV
if st.session_state.chat_history:
    if st.button("Export Chat History to CSV"):
        hist_df = pd.DataFrame(st.session_state.chat_history)
        hist_df.to_csv("chat_history.csv", index=False)
        st.success("Chat history saved to chat_history.csv")