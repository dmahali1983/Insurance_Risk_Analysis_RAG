import pinecone

def setup_pinecone(api_key, index_name):
    pinecone.init(api_key=api_key, environment='us-west1-gcp')
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=384)
    return pinecone.Index(index_name)