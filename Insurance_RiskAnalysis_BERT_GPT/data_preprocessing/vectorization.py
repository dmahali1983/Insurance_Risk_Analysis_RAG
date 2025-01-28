from sentence_transformers import SentenceTransformer

def vectorize_text(df, column='description'):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df['vector'] = df[column].apply(lambda x: model.encode(x))
    return df