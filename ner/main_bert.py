# import os
# import pytesseract
# from PIL import Image
# import cv2
# import numpy as np
# from pdf2image import convert_from_path
# import re
# import json
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForTokenClassification
# import torch

# # Path to Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # Set TESSDATA_PREFIX environment variable to the correct path
# os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

# # Path to Poppler binary
# poppler_path = r"C:\poppler-24.02.0\Library\bin"

# def detect_header_footer_height(img, threshold=200, header_search_height=200, footer_search_height=200):
#     img_np = np.array(img)
#     gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     header_height = 50
#     footer_height = 120
#     img_height = img_np.shape[0]    

#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         if y < header_search_height and y + h < header_search_height:
#             if y + h > header_height:
#                 header_height = y + h
#         if y > img_height - footer_search_height:
#             if h > footer_height:
#                 footer_height = h
    
#     return header_height, footer_height

# def extract_text_from_pdf(pdf_path, poppler_path):
#     images = convert_from_path(pdf_path, poppler_path=poppler_path)
#     text = ""
#     for img in images:
#         try:
#             header_height, footer_height = detect_header_footer_height(img)
#             width, height = img.size
#             img_cropped = img.crop((0, header_height, width, height - footer_height))
#             img_grayscale = img_cropped.convert('L')
#             text += pytesseract.image_to_string(img_grayscale, lang='ind')
#         except pytesseract.TesseractError as e:
#             print(f"Tesseract error: {e}")
#             raise
#     return text

# def preprocess_text(text):
#     text = re.sub(r'[^\w\s]', '', text)
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()

# def split_text(text, max_length=512):
#     tokens = tokenizer.tokenize(text)
#     chunks = []
#     current_chunk = []

#     for token in tokens:
#         if len(current_chunk) + 1 > max_length:
#             chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
#             current_chunk = []
#         current_chunk.append(token)

#     if current_chunk:
#         chunks.append(tokenizer.convert_tokens_to_string(current_chunk))

#     return chunks

# def run_ner(text, tokenizer, model, max_length=512):
#     ner_results = []
#     chunks = split_text(text, max_length)
    
#     for chunk in chunks:
#         inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=max_length)
#         with torch.no_grad():
#             outputs = model(**inputs)
        
#         logits = outputs.logits
#         predictions = torch.argmax(logits, dim=2)
#         tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
#         labels = [model.config.id2label[p.item()] for p in predictions[0]]
        
#         for token, label in zip(tokens, labels):
#             ner_results.append({"word": token, "entity": label})
    
#     return ner_results

# def convert_to_dataset(text, ner_results):
#     entities = {
#         "who": [],
#         "when": [],
#         "where": [],
#         "what": [],
#         "how_much": []
#     }

#     for entity in ner_results:
#         word = entity['word']
#         label = entity['entity']
#         start = text.find(word)
#         end = start + len(word)

#         entity_data = {"word": word, "start": start, "end": end}
        
#         if "PER" in label:
#             entities["who"].append(entity_data)
#         elif "DATE" in label or "TIME" in label:
#             entities["when"].append(entity_data)
#         elif "LOC" in label or "ORG" in label:
#             entities["where"].append(entity_data)
#         elif "MISC" in label:
#             entities["what"].append(entity_data)
#         elif "MONEY" in label or "QUANTITY" in label:
#             entities["how_much"].append(entity_data)

#     for key in entities:
#         entities[key] = sorted(list({(e['start'], e['end'], e['word']): e for e in entities[key]}.values()), key=lambda x: x['start'])
    
#     dataset = {
#         "text": text,
#         "entities": entities
#     }
    
#     return dataset

# def convert_to_bio_tagging(text, ner_results):
#     tokens = tokenizer.tokenize(text)
#     bio_tagging = ["O"] * len(tokens)
    
#     # Create a map from token index to NER entity
#     token_to_entity = {}
#     for entity in ner_results:
#         entity_tokens = tokenizer.tokenize(entity['word'])
#         start_idx = len(tokenizer.tokenize(text[:text.find(entity['word'])]))
#         end_idx = start_idx + len(entity_tokens)
        
#         for i in range(start_idx, end_idx):
#             if i < len(bio_tagging):
#                 if i == start_idx:
#                     bio_tagging[i] = "B-" + entity['entity']
#                 else:
#                     bio_tagging[i] = "I-" + entity['entity']
    
#     return list(zip(tokens, bio_tagging))

# # Load BERT-based NER model from Hugging Face
# model_name = "indolem/indobert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForTokenClassification.from_pretrained(model_name)
# model.eval()

# folder_path = r"C:\Users\Asus\Documents\bert\ner\Dokumen"
# pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
# all_bio_tagging = []

# for pdf_file in pdf_files:
#     pdf_path = os.path.join(folder_path, pdf_file)
#     text = extract_text_from_pdf(pdf_path, poppler_path)
#     cleaned_text = preprocess_text(text)
#     ner_results = run_ner(cleaned_text, tokenizer, model)
#     dataset = convert_to_dataset(cleaned_text, ner_results)
#     bio_tagging = convert_to_bio_tagging(cleaned_text, ner_results)
#     all_bio_tagging.extend(bio_tagging)
    
#     output_file = os.path.splitext(pdf_file)[0] + "_ner_dataset.json"
#     output_path = os.path.join(folder_path, output_file)
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(dataset, f, ensure_ascii=False, indent=4)
    
#     print(f"Dataset saved to {output_path}")

# df_bio = pd.DataFrame(all_bio_tagging, columns=["Token", "BIO_Tag"])
# bio_csv_output_path = os.path.join(folder_path, 'bio_tagging.csv')
# df_bio.to_csv(bio_csv_output_path, index=False, encoding='utf-8')
# print(f"BIO tagging saved to {bio_csv_output_path}")
