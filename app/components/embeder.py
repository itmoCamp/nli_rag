def embed_doc(model, text: str) -> list[float]:
    doc_embedding = model.run(text)
    return doc_embedding
