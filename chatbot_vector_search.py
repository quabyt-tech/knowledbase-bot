import gradio as gr
import chromadb
from chromadb.utils import embedding_functions
import constants

persist_dir = constants.CHROMA_PERSIST_DIR
collection_name = constants.CHROMA_COLLECTION_NAME


def get_store_response(message) -> str:
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L12-v2"  # Chroma uses all-MiniLM-L6-v2 by default. Another good choice: e5-small-v2
    )

    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection = chroma_client.get_collection(
        name=collection_name, embedding_function=sentence_transformer_ef
    )  # use the same embedding function that was used when docements were saved

    results = collection.query(query_texts=[f"{message}"], n_results=1)

    if results["documents"]:
        all_documents = ""
        for i, doc in enumerate(results["documents"]):
            doc_text = f"\n\nNEW DOCUMENT\n\n".join(doc)
            all_documents += doc_text + "\n\n"
        return all_documents
    else:
        return "No stored response found"


def predict(message, history):
    gpt_response = get_store_response(message)
    return gpt_response


gr.ChatInterface(
    predict,
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="Ask me a question", container=True, scale=7),
    title="KB Bot",
    description="Chat with the vector database",
    theme="default",
    examples=[
        "Tell me about swimlanes in architecture guidelines",
        "How to invoice accounting?",
        "What is the open door policy?",
    ],
    cache_examples=True,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
).launch()
