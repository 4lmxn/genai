import gensim.downloader as api
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

wv = api.load("glove-wiki-gigaword-100")

words = ["king", "queen", "man", "woman", "prince", "princess"]
vectors = [wv[w] for w in words]

pca = PCA(n_components=2)
result = pca.fit_transform(vectors)

plt.figure(figsize=(8, 6))
for i, w in enumerate(words):
    plt.scatter(result[i, 0], result[i, 1])
    plt.text(result[i, 0], result[i, 1], w)

plt.title("Word Embeddings (PCA)")
plt.show()

print("\nSimilar words to 'king':")
print(wv.most_similar("king", topn=5))
