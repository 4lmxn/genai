import gensim.downloader as api
wv = api.load("glove-wiki-gigaword-100")

def explore(a, b, c):
    print(f"\n{a} - {b} + {c}:")
    result = wv[a] - wv[b] + wv[c]
    for w, s in wv.similar_by_vector(result, topn=5):
        if w not in [a, b, c]:
            print(w, round(s, 4))

explore("king", "man", "woman")
explore("paris", "france", "germany")
explore("apple", "fruit", "carrot")

print("\nSimilarity:", round(wv.similarity("cat", "dog"), 4))
print("Similarity:", round(wv.similarity("computer", "keyboard"), 4))

for word in ["happy", "sad", "technology"]:
    print(f"\nSimilar to {word}:")
    for w, s in wv.most_similar(word, topn=5):
        print(w, round(s, 4))
