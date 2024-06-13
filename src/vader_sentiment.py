from config import DEVELOPER_KEY
import googleapiclient.discovery
import gradio as gr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sqlite3

conn = sqlite3.connect('youtube_comments.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS comments
             (author text, sentiment text, comment text)''')
conn.commit()

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = DEVELOPER_KEY
analyzer = SentimentIntensityAnalyzer()

def analyze_comments(video_id, max_results):
    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results
    )
    response = request.execute()

    comments = []

    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        text = comment['textDisplay']

        vs = analyzer.polarity_scores(text)
        compound_score = vs['compound']

        if compound_score >= 0.05:
            sentiment_category = "Positive"
        elif compound_score <= -0.05:
            sentiment_category = "Negative"
        else:
            sentiment_category = "Neutral"

        comments.append([
            comment['authorDisplayName'],
            sentiment_category,
            comment['textDisplay'],
        ])

        c.execute("INSERT INTO comments (author, sentiment, comment) VALUES (?, ?, ?)",
                  (comment['authorDisplayName'], sentiment_category, comment['textDisplay']))

    conn.commit()

    df = gr.DataFrame(
        comments,
        headers=["Author", "Sentiment", "Comment"]
    )

    return df

def fetch_comments():
    c.execute("SELECT * FROM comments")
    all_comments = c.fetchall()
    for comment in all_comments:
        print(comment)

with gr.Blocks() as iface:
    gr.Markdown("YouTube Comment Sentiment Analysis - VADER")
    gr.Markdown("Enter a YouTube Video ID and select the number of comments to analyze their sentiments.")
    with gr.Row():
        video_id_input = gr.Textbox(label="YouTube Video ID", placeholder="Enter YouTube Video ID here...")
        max_results_input = gr.Slider(minimum=1, maximum=200, label="Number of Comments", step=1)
        submit_button = gr.Button("Check comment sentiment")
        fetch_button = gr.Button("Fetch All Comments from DB")
    output_area = gr.DataFrame(
        label="Comments and Sentiments",
        headers=["Author", "Sentiment", "Comment"],
    )

    submit_button.click(analyze_comments, inputs=[video_id_input, max_results_input], outputs=output_area)
    fetch_button.click(fetch_comments)

iface.launch()
conn.close()