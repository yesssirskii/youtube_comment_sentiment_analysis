from config import DEVELOPER_KEY
import googleapiclient.discovery
import gradio as gr
import sqlite3
from transformers import pipeline

model_name = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_pipeline = pipeline("text-classification", model=model_name, top_k=1)

label_mapping = {
    'LABEL_0': 'Negative',
    'LABEL_1': 'Neutral',
    'LABEL_2': 'Positive'
}

conn = sqlite3.connect('youtube_comments.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS comments
             (author text, sentiment text, comment text)''')
conn.commit()

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = DEVELOPER_KEY

def analyze_comments(video_id, max_results):
    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)
    request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=max_results)
    response = request.execute()
    comments = []

    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        text = comment['textDisplay']
        results = sentiment_pipeline(text)
        sentiment_category = label_mapping[results[0][0]['label']]
        comments.append([comment['authorDisplayName'], sentiment_category, comment['textDisplay']])
        c.execute("INSERT INTO comments (author, sentiment, comment) VALUES (?, ?, ?)",
                  (comment['authorDisplayName'], sentiment_category, comment['textDisplay']))
    conn.commit()
    return gr.DataFrame(comments, headers=["Author", "Sentiment", "Comment"])

def fetch_comments():
    conn = sqlite3.connect('youtube_comments.db')
    c = conn.cursor()
    c.execute("SELECT * FROM comments")
    all_comments = c.fetchall()
    for comment in all_comments:
        print(comment)
    conn.close()

with gr.Blocks() as iface:
    gr.Markdown("YouTube Comment Sentiment Analysis - RoBERTa")
    gr.Markdown("Enter a YouTube Video ID and select the number of comments to analyze their sentiments.")
    with gr.Row():
        video_id_input = gr.Textbox(label="YouTube Video ID", placeholder="Enter YouTube Video ID here...")
        max_results_input = gr.Slider(minimum=1, maximum=200, label="Number of Comments", step=1)
        submit_button = gr.Button("Check comment sentiment")
        fetch_button = gr.Button("Fetch All Comments From DB")
    output_area = gr.DataFrame(label="Comments and Sentiments", headers=["Author", "Sentiment", "Comment"])
    submit_button.click(analyze_comments, inputs=[video_id_input, max_results_input], outputs=output_area)
    fetch_button.click(fetch_comments)

iface.launch()

conn.close()