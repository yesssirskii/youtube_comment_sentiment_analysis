###### Autor: Ivan Matejčić
###### Datum: 09.06.2024.

# YouTube Comment Sentiment Analysis: Tehnička Dokumentacija

## Uvod

Ovaj dokument pruža tehnički pregled projekta za analizu sentimenata YouTube komentara.  
Projekt je dizajniran da preuzme komentare s određenog YouTube videa i analizira njihove sentimente koristeći prethodno trenirani model za analizu sentimenata (VADER ili RoBERTa) te ih potom spremi u SQLite bazu podataka. Rezultat se zatim prikazuje u engl. *user-friendly* Gradio korisničkom sučelju.

## 1. Upute za korištenje

### Instalacija potrebnih paketa
Za pokretanje projekta, potrebni su slijedeći paketi:

- Python 3.x
- Gradio
- googleapiclient
- transformers
- SQLite3

```bash
pip3 install gradio google-api-python-client transformers sqlite3
```

### Google YouTube API Key

Za pokretanje projekta potreban je validan API key koji je potrebno upisati u *config.py* datoteku.
Upute za navedeno možete pregledati [ovdje](https://docs.themeum.com/tutor-lms/tutorials/get-youtube-api-key/).

### Pokretanje projekta

Za pokretanje projekta, potrebno je pokrenuti slijedeću naredbu  
(ovisno koji model želimo korisiti):
```bash
python3 vader_sentiment.py
```

```bash
python3 roberta_sentiment.py
```

<span style="color:red; font-weight: bold">NAPOMENA</span>: kod pokretanja RoBERTa projekta, moguće je čekanje dohvaćanja samog modela:

## 2. Priprema Podataka

### Pregled
Aplikacija dohvaća komentare s YouTube videozapisa koristeći YouTube Data API.  
API zahtijeva ključ API-ja i ID videozapisa za dohvaćanje komentara.

### Implementacija Koda
Google API klijentska biblioteka se koristi za interakciju s YouTube Data API-jem.  

```python
import googleapiclient.discovery

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "DEVELOPER_KEY"
```

## 3. Dohvaćanje podataka

### Pregled i implementacija koda

Funkcija **analyze_comments** se koristi za dohvaćanje komentara.  
Uzima ID videozapisa i maksimalni broj komentara koji će se dohvatiti kao parametre.  
Pošto je implementacija drukčija za VADER i RoBERTa model, ova funkcija ima drukčiju strukturu koda za svaki model, no funkcionalnost je jednaka.

```python
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
```

## 4. Rješenje ML problema

### Pregled

Projekt koristi dva jezična modela za analizu sentimenta: **VADER** (Valence Aware Dictionary and sEntiment Reasoner) SentimentIntensityAnalyzer i **cardiffnlp/twitter-roberta-base-sentiment** modele za analizu sentimenata.  

**VADER** je leksikonski i pravilima zasnovan alat i engl. *pretrained* model za analizu sentimenata posebno usklađen za sentimente izražene na društvenim mrežama.  
**cardiffnlp/twitter-roberta-base-sentiment** je model baziran na RoBERTa modelu te je izmijenjen i dopunjen na način da su njegovi podaci bazirani na podacima sa Twitter-a (komentari, objave i slično).

Ovakvi modeli su nam najpovoljniji za ovaj projekt zato jer se u našem slučaju radi sa komentarima sa društvenih mreža, za što su ovi modeli napravljeni.

### Analiza Sentimenata VADER modelom

Metoda **polarity_scores** od VADER-a se koristi za analizu sentimenta svakog komentara.  
Složeni rezultat se zatim koristi za klasificiranje sentimenta kao pozitivnog, negativnog ili neutralnog.

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

for item in response['items']:
    text = item['snippet']['topLevelComment']['snippet']['textDisplay']
    vs = analyzer.polarity_scores(text)
    compound_score = vs['compound']
```

### Analiza Sentimenata RoBERTa modelom

Analiza sentimenta RoBERT-a modelom započinje definiranjem modela i sentiment pipeline-a.

```python
from transformers import pipeline

model_name = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_pipeline = pipeline("text-classification", model=model_name, top_k=1)
```
Kasnije se ovi podaci pozovu u **analyze_comments** funkciji i spremaju u varijablu **results**:

```python
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
```

## 5. Implementacija korisničkog sučelja

### Pregled

Projekt koristi **Gradio**, open-source Python biblioteku, za stvaranje interaktivnog web sučelja.

### Izgradnja Sučelja

Sučelje je samo po sebi jednostavno; sastoji se od tekstualnog tekstualnog polja za video ID, klizača za odabir broja komentara, gumba za pokretanje analize, gumba za prikaz svih komentara iz baze podataka i tablice sa dohvaćenim komentarima, imenom korisnika i sentimentom komentara.
Analizirani komentari se prikazuju u Gradio DataFrame-u.

```python
import gradio as gr

with gr.Blocks() as iface:
    gr.Markdown("YouTube Comment Sentiment Analysis")
    gr.Markdown("Enter a YouTube Video ID and select the number of comments to analyze their sentiments.")
    with gr.Row():
        video_id_input = gr.Textbox(label="YouTube Video ID", placeholder="Enter YouTube Video ID here...")
        max_results_input = gr.Slider(minimum=1, maximum=200, label="Number of Comments", step=1)
        submit_button = gr.Button("Check comment sentiment")
        fetch_button = gr.Button("Fetch All Comments From DB")
    output_area = gr.DataFrame(label="Comments and Sentiments", headers=["Author", "Sentiment", "Comment"])
    submit_button.click(analyze_comments, inputs=[video_id_input, max_results_input], outputs=output_area)
    fetch_button.click(fetch_comments)
```

## 6. Spremanje podataka

Za spremanje podataka koristi se SQLite baza podataka radi jednostavne instalacije i implementacije.

Potrebno je instalirati sqlite3 paket:
```bash
 pip3 install sqlite3
 ```

Potom ga importati:
``` python
import sqlite3 
```

Te postaviti vezu sa bazom i kreirati novu tablicu koju smo u našem slučaju nazvali **youtube_comments** sa kolonama   *autohr*, *sentimet* i *comment*.

```python
conn = sqlite3.connect('youtube_comments.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS comments
             (author text, sentiment text, comment text)''')
conn.commit()
```

U **analyze_comments** funkciji nakon što se dohvate komentari, spremamo ih u bazu:
```python
c.execute("INSERT INTO comments (author, sentiment, comment) VALUES (?, ?, ?)",
            (comment['authorDisplayName'], sentiment_category, comment['textDisplay']))
conn.commit()
```

Naposlijetku, definirana je funkcija **fetch_comments* koja pokrene SQL query za dohvat svih komentara iz baze podataka i izlista ih u terminalu:
```python
def fetch_comments():
    conn = sqlite3.connect('youtube_comments.db')
    c = conn.cursor()
    c.execute("SELECT * FROM comments")
    all_comments = c.fetchall()
    for comment in all_comments:
        print(comment)
    conn.close()
```

Ova funkcija se poziva kada se klikne na gumb "Fetch Comments From DB" u Gradio sučelju:
```python
fetch_button.click(fetch_comments)
```

## 7. Pretpregled

Izgledi sučelja su identični za oba modela.

### VADER

![VADER preview](/src/img/vader_preview.png)

### RoBERTa

![RoBERTa preview](/src//img/roberta_preview.png)

## 8. Literatura

https://www.gradio.app/guides/sharing-your-app#hosting-on-hf-spaces
https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment
https://github.com/cjhutto/vaderSentiment