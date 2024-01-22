from config import DEVELOPER_KEY

import googleapiclient.discovery
import googleapiclient.errors
import googleapiclient.discovery
import gradio as gr

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = DEVELOPER_KEY

analyzer = SentimentIntensityAnalyzer()

def analyze_comments(video_id, max_results):
    """Get comments from a video using video_id and show them and their sentiment as a DataFrame"""

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)

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

         # Analyze sentiment using VADER
        vs = analyzer.polarity_scores(text)
        compound_score = vs['compound']

        if compound_score >= 0.05:
            sentiment_category = "Positive"
        elif compound_score <= -0.05:
            sentiment_category = "Negative"
        else:
            sentiment_category = "Neutral"

        comment = item['snippet']['topLevelComment']['snippet']
        comments.append([
            comment['authorDisplayName'],
            sentiment_category,
            comment['textDisplay'],
        ])

    df = gr.DataFrame(
        comments,
        {
        "Author" : ['author'],
        "Sentiment" : ['sentiment_category'],
        "Comment" : ['textDisplay'],
        }
    )

    return df

with gr.Blocks() as iface:
    gr.Markdown("YouTube Comment Sentiment Analysis")
    gr.Markdown("Enter a YouTube Video ID and select the number of comments to analyze their sentiments.")
    with gr.Row():
        video_id_input = gr.Textbox(label="YouTube Video ID", placeholder="Enter YouTube Video ID here...")
        max_results_input = gr.Slider(minimum=1, maximum=200, label="Number of Comments", step=1)
        submit_button = gr.Button("Check comment sentiment")
    output_area = gr.DataFrame(
        label="Comments and Sentiments",
        headers=["User", "Sentiment", "Comment"],
        )

    submit_button.click(analyze_comments, inputs=[video_id_input, max_results_input], outputs=output_area)

iface.launch()