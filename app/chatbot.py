from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def ask_question(question):

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory="models/chroma_db",
        embedding_function=embedding_model
    )

    docs = db.similarity_search(question, k=1)

    if len(docs) == 0:
        return "No relevant information found in uploaded notes."

    response = "📘 Answer From Notes:\n"

    for doc in docs:
        response += doc.page_content[:500] + "\n"

    return response