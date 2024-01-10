from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import re
import numpy as np
import tensorflow_hub as hub
from sklearn.neighbors import NearestNeighbors
import openai
from flask import Flask
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)


class SemanticSearch:
    
    def __init__(self):
        self.use = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        self.fitted = False
    
    def fit(self, data, batch=1000, n_neighbors=5):
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch=batch)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True
    
    def __call__(self, text, return_data=True):
        inp_emb = self.use([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]
        
        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors
    
    def get_text_embedding(self, texts, batch=1000):
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i:(i+batch)]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings


def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text


def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page-1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list


def text_to_chunks(texts, word_length=150, start_page=1):
    text_toks = [t.split(' ') for t in texts]
    page_nums = []
    chunks = []
    
    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i:i+word_length]
            if (i+word_length) > len(words) and (len(chunk) < word_length) and (
                len(text_toks) != (idx+1)):
                text_toks[idx+1] = chunk + text_toks[idx+1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[Página nº. {idx+start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


recommender = SemanticSearch()


def extract_from_pdf(keywords, file_path):
    print(f"Keywords: {keywords}, File Path: {file_path}")
    texts = pdf_to_text(file_path)
    chunks = text_to_chunks(texts)
    recommender.fit(chunks)
    topn_chunks = recommender(keywords[0])  # Assume que keywords é uma lista e usa o primeiro termo para a busca
    extracted_text = "\n".join(topn_chunks)
    return extracted_text


logging.basicConfig(level=logging.DEBUG)

@app.route('/search_pdf', methods=['POST'])
def search_pdf():
    logging.debug(f"Received request: {request.json}")
    keywords = request.json.get('keywords')
    file_path = "/Users/anderson/Documents/Meus projetos/Backup light/26. light 2 (ultimo publicado)/14.133:2021.pdf"
    extracted_text = extract_from_pdf(keywords, file_path)
    print(extracted_text)
    return jsonify({'trechos': extracted_text})


@app.route("/")
def index():
    return "Hello, World!"


if __name__ == '__main__':
    app.run(port=5000, debug=True)
