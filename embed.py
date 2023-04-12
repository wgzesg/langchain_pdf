from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np
import flask

# def cosine_similarity(a, b):
#     # cosince similarity
#     print(len(a), len(b))
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
hf = HuggingFaceEmbeddings(model_name=model_name)


app = flask.Flask(__name__)

@app.route('/embed', methods=['POST'])
def embed():
    data = flask.request.json
    print(data)
    return hf.embed_query(data['query'])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)


