from gensim.models import Word2Vec

corpus = [
    "patient has fever",
    "doctor gives treatment",
    "medicine helps recovery",
    "virus causes infection"
]

sentences = [s.split() for s in corpus]
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1)

print("\nSimilar to 'treatment':")
print(model.wv.most_similar("treatment"))

print("\nSimilar to 'virus':")
print(model.wv.most_similar("virus"))
