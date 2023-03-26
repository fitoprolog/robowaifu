from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

sentence1 = "cerca de la casa habia un perro"

while True:
  sentence2= input(">")
  if sentence2=="quit": break
  embeddings = model.encode([sentence1, sentence2])
  cosine_similarity = util.cos_sim(embeddings[0], embeddings[1])
  print(cosine_similarity)

