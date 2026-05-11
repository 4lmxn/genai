from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

with open("sample_text.txt", "r", encoding="utf-8") as file:
    text_content = file.read()

template = """
Document: {text}

Summary: Provide a concise summary.
Key Takeaways: List 3 important points.
Sentiment: Is the document Positive, Negative, or Neutral?
"""

prompt = template.format(text=text_content)
summary = summarizer(prompt, max_length=200, min_length=50)

print(summary[0]['summary_text'])
