###### Autor: Ivan Matejčić

# YouTube Comment Sentiment Analysis: Tehnička Dokumentacija

Ovaj dokument pruža tehnički pregled aplikacije za analizu sentimenata YouTube komentara. Aplikacija je dizajnirana da preuzme komentare s određenog YouTube videa i analizira njihove sentimente koristeći prethodno trenirani model za analizu sentimenata (VADER). Rezultat se zatim prikazuje u engl. *user-friendly* Gradio korisničkom sučelju.

## 1. Priprema Podataka

### Pregled
Aplikacija dohvaća komentare s YouTube videozapisa koristeći YouTube Data API.  
API zahtijeva ključ API-ja i ID videozapisa za dohvaćanje komentara.

### Implementacija Koda
Google API klijentska biblioteka se koristi za interakciju s YouTube Data API-jem.  
Evo relevantnog isječka koda:

```python
import googleapiclient.discovery

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "DEVELOPER_KEY"

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)
```

### Dohvaćanje podataka

Funkcija **analyze_comments** se koristi za dohvaćanje komentara.  
Uzima ID videozapisa i maksimalni broj komentara koji će se dohvatiti kao parametre.

```python
def analyze_comments(video_id, max_results):
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results
    )
    response = request.execute()
```

## 2. Rješenje ML problema

###Pregled

Aplikacija koristi VADER (Valence Aware Dictionary and sEntiment Reasoner) SentimentIntensityAnalyzer za analizu sentimenata.  
VADER je leksikonski i pravilima zasnovan alat i engl. *pretrained* model za analizu sentimenata posebno usklađen za sentimente izražene na društvenim mrežama.  
Ovakav model nam je najprikladniji zato jer se u ovom slučaju radi sa komentarima sa društvenih mreža.

###Analiza Sentimenata s VADER-om

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

## 3. Implementacija korisničkog sučelja

### Pregled
Aplikacija koristi Gradio, open-source Python biblioteku, za stvaranje interaktivnog web sučelja.

### Izgradnja Sučelja
Sučelje je samo po sebi jednostavno; sastoji se od tekstualnog tekstualnog polja za video ID, klizača za odabir broja komentara i gumba za pokretanje analize.  
Analizirani komentari se prikazuju u Gradio DataFrame-u.

```python
import gradio as gr

with gr.Blocks() as iface:
    video_id_input = gr.Textbox(label="YouTube Video ID")
    max_results_input = gr.Slider(minimum=1, maximum=200, label="Broj Komentara", step=1)
    submit_button = gr.Button("Provjeri sentiment komentara")
    output_area = gr.DataFrame(label="Komentari i Sentimenti")

    submit_button.click(analyze_comments, inputs=[video_id_input, max_results_input], outputs=output_area)
```

## 4. Pretpregled

![app_preview](/images/app_preview.png)

## 5. Hosting

Za hosting python skripte koristi se [HuggingFace Spaces](https://huggingface.co/spaces).  
HuggingFace Spaces se koristi zato jer specifično pruža hosting Gradio aplikacija, što je potrebno u ovom slučaju.

## 6. Literatura

https://www.gradio.app/guides/sharing-your-app#hosting-on-hf-spaces
https://huggingface.co/spaces