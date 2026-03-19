from elasticsearch import Elasticsearch

# Connect to Elasticsearch
es = Elasticsearch(hosts=["http://localhost:9200"])

# Check connection
es.ping()

# Create an index
index_name = "api"

if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)

if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name)

# Add documents
documents = [
    {"title": "Starting", "content": "This is the first api message."}
]

for i, doc in enumerate(documents):
    # Use 'body' instead of 'document'
    es.index(index=index_name, id=i + 1, body=doc)

# Create an index
index_name = "streamlit"

if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)

if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name)

# Add documents
documents = [
    {"title": "Starting", "content": "This is the streamlit message."}
]

for i, doc in enumerate(documents):
    # Use 'body' instead of 'document'
    es.index(index=index_name, id=i + 1, body=doc)
