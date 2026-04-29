import gensim.downloader as api
import random

wv = api.load("glove-wiki-gigaword-100")

def generate(seed):
    sim = [w for w, _ in wv.most_similar(seed, topn=5)]
    sentences = [
        f"The {seed} is surrounded by {sim[0]} and {sim[1]}.",
        f"People relate {seed} with {sim[2]} and {sim[3]}.",
        f"In the world of {seed}, {sim[4]} is common."
    ]
    return " ".join(sentences)

word = input("Enter word: ")
print("\nGenerated Paragraph:\n")
print(generate(word))
