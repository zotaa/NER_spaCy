import os
import pytesseract
from PIL import Image
import cv2
import numpy as np
from pdf2image import convert_from_path
import spacy
from spacy.training.example import Example
import re
import json
import random
import pandas as pd
from collections import Counter
from heapq import nlargest
from string import punctuation

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set TESSDATA_PREFIX environment variable to the correct path
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# Path to Poppler binary
poppler_path = r"C:\poppler-24.02.0\Library\bin"

def detect_header_footer_height(img, threshold=200, header_search_height=200, footer_search_height=200, footer_extra_height=450):
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    header_height = 50
    footer_height = 130
    img_height = img_np.shape[0]    

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if y < header_search_height and y + h < header_search_height:
            if y + h > header_height:
                header_height = y + h
        if y > img_height - footer_search_height:
            if y + h > img_height - footer_height:
                footer_height = img_height - y  # Dynamically set footer height based on position

    footer_height += footer_extra_height  # Increase the footer height by a specified extra amount
    return header_height, footer_height

def extract_text_from_pdf(pdf_path, poppler_path):
    # Convert PDF to images and extract text using OCR
    images = convert_from_path(pdf_path, poppler_path=poppler_path)
    text = ""
    for img in images:
        try:
            header_height, footer_height = detect_header_footer_height(img)
            width, height = img.size
            img_cropped = img.crop((0, header_height, width, height - footer_height))
            img_grayscale = img_cropped.convert('L')
            text += pytesseract.image_to_string(img_grayscale, lang='ind')
        except pytesseract.TesseractError as e:
            print(f"Tesseract error: {e}")
            raise
    return text

def preprocess_text(text):
    # Basic preprocessing to clean the text
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_train_data(json_path): 
    # Load the training data from a JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    return train_data

def train_spacy_model(train_data, output_dir, n_iter=100):
    # Create a blank spaCy model for the Indonesian language
    nlp = spacy.blank("id")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)

    # Add sentencizer to the pipeline
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    # Add labels to the NER pipeline
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    optimizer = nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(train_data)
        losses = {}
        for text, annotations in train_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.5, losses=losses)
        print(f"Iteration {itn + 1}/{n_iter}, Losses: {losses}")

    nlp.to_disk(output_dir)
    print(f"Model saved to {output_dir}")

def run_ner(text, nlp):
    # Run named entity recognition on the text
    doc = nlp(text)
    ner_results = []
    for ent in doc.ents:
        ner_results.append({'word': ent.text, 'entity': ent.label_, 'start': ent.start_char, 'end': ent.end_char})
    return ner_results

def convert_to_dataset(text, ner_results):
    # Convert NER results to a dataset format
    entities = {
        "who": [],
        "when": [],
        "where": [],
        "what": [],
        "how_much": []
    }

    for entity in ner_results:
        word = entity['word']
        label = entity['entity']
        start = entity['start']
        end = entity['end']

        entity_data = {"word": word, "start": start, "end": end}
        
        if "PER" in label:
            entities["who"].append(entity_data)
        elif "DATE" in label or "TIME" in label:
            entities["when"].append(entity_data)
        elif "LOC" in label or "ORG" in label:
            entities["where"].append(entity_data)
        elif "MISC" in label:
            entities["what"].append(entity_data)
        elif "MONEY" in label or "QUANTITY" in label:
            entities["how_much"].append(entity_data)

    # Ensure entities are unique and sorted
    for key in entities:
        entities[key] = sorted(list({(e['start'], e['end'], e['word']): e for e in entities[key]}.values()), key=lambda x: x['start'])
    
    dataset = {
        "text": text,
        "entities": entities
    }
    
    return dataset

def convert_to_bio_tagging(text, ner_results):
    # Convert NER results to BIO tagging format
    tokens = text.split()
    bio_tagging = ["O"] * len(tokens)
    for entity in ner_results:
        entity_tokens = entity['word'].split()
        start_idx = text.find(entity['word'])
        end_idx = start_idx + len(entity['word'])
        token_start_idx = len(text[:start_idx].split())
        token_end_idx = token_start_idx + len(entity_tokens)
        bio_tagging[token_start_idx] = "B-" + entity['entity']
        for i in range(token_start_idx + 1, token_end_idx):
            bio_tagging[i] = "I-" + entity['entity']
    return list(zip(tokens, bio_tagging))

def summarize_with_entities(text, entities, num_sentences=1):
    # Load the text and split it into sentences
    nlp = spacy.blank("id")
    sentencizer = nlp.add_pipe("sentencizer")
    doc = nlp(text)
    sentences = list(doc.sents)

    # Preprocess sentences
    sentence_texts = [sent.text.strip().lower().translate(str.maketrans('', '', punctuation)) for sent in sentences]

    # Word frequency calculation (excluding stopwords)
    stopwords = set(nlp.Defaults.stop_words)
    word_frequencies = Counter()
    for sentence in sentence_texts:
        for word in sentence.split():
            if word not in stopwords:
                word_frequencies[word] += 1

    # Normalize word frequencies
    max_frequency = max(word_frequencies.values(), default=1)
    for word in word_frequencies:
        word_frequencies[word] /= max_frequency

    # Count entity occurrences in the text
    entity_counts = Counter(entities)

    # Define priority entities
    priority_entities = ["who", "where", "when"]

    # Calculate the score for each sentence based on entity presence and word frequency
    sentence_scores = {}
    for sent in sentences:
        sentence_score = 0
        sentence_word_count = len(sent.text.split())

        for word in sent.text.lower().split():
            if word in word_frequencies:
                sentence_score += word_frequencies[word]

        for ent in entities:
            if ent in sent.text:
                # Give more weight to priority entities
                weight = 3.0 if ent in priority_entities else 1.5
                sentence_score += entity_counts[ent] * weight

        # Adjust score for sentence length
        if sentence_word_count > 0:
            sentence_score /= sentence_word_count

        sentence_scores[sent] = sentence_score

    # Filter out sentences with low scores
    average_score = sum(sentence_scores.values()) / len(sentence_scores) if sentence_scores else 0
    filtered_sentences = {sent: score for sent, score in sentence_scores.items() if score > average_score}

    # Select sentences with the highest scores
    summary_sentences = nlargest(num_sentences, filtered_sentences, key=filtered_sentences.get)

    # Combine sentences to form the summary
    summary = ' '.join([sent.text for sent in summary_sentences])
    return summary

folder_path = r"C:\Users\Asus\Documents\bert\ner\Dokumen"
pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
nlp_model_path = r"C:\Users\Asus\Documents\bert\ner\model"

# Load or train spaCy model
if os.path.exists(nlp_model_path):
    nlp = spacy.load(nlp_model_path)
else:
    json_path = r"C:\Users\Asus\Documents\bert\ner\train_data.json"
    train_data = load_train_data(json_path)
    train_spacy_model(train_data, nlp_model_path)
    nlp = spacy.load(nlp_model_path)

for pdf_file in pdf_files:
    pdf_path = os.path.join(folder_path, pdf_file)
    
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path, poppler_path)
    print(f"Extracted Text from {pdf_file}:")
    print(text)
    
