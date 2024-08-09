from transformers import AutoTokenizer, AutoModelForTokenClassification

# Tentukan nama model
model_name = "cahya/bert-base-indonesian-NER"

# Unduh tokenizer dan model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

print("Model dan tokenizer berhasil diunduh dan diinisialisasi.")
