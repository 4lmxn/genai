import gensim.downloader as api
from transformers import pipeline

wv = api.load("glove-wiki-gigaword-100")
generator = pipeline("text-generation", model="gpt2")

def enrich(prompt, word):
    similar = wv.most_similar(word, topn=1)[0][0]
    return prompt.replace(word, similar)

prompt = "Who is king"
new_prompt = enrich(prompt, "king")

print("\nOriginal Prompt:")
print(generator(prompt, max_length=50)[0]['generated_text'])

print("\nEnhanced Prompt:")
print(generator(new_prompt, max_length=50)[0]['generated_text'])
