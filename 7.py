from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """Artificial Intelligence and its impact on modern society.
Artificial Intelligence is a transformative branch of computer science dedicated to
creating systems capable of performing tasks that typically require human intelligence
such as reasoning, problem solving and understanding language."""

summary = summarizer(text, max_length=300, min_length=30)

print("Original text:\n", text)
print("\nSummarized text:\n", summary[0]['summary_text'])
