from sentence_transformers import SentenceTransformer, util
import numpy as np
def read_lines_from_files(file_paths):
    all_lines = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            all_lines.extend([line.strip() for line in lines])
    return all_lines


train_paths = [ "dialogue_loan.txt", "dialogue_org.txt"]

# Corpus with example sentences
corpus = read_lines_from_files(train_paths)
#print(corpus)
embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Query sentences
while True : 
    query = input("문장 : ")
    if query == 'q' :
        break
    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = 5
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu()
    
    #We use np.argpartition, to only partially sort the top_k results
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
    
    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")
    
    for idx in top_results[0:top_k]:
        print(corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))
         