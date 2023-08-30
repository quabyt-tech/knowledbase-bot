import os
import chromadb
from chromadb.utils import embedding_functions
import constants
import gradio
import cohere
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()
persist_dir = constants.CHROMA_PERSIST_DIR
collection_name = constants.CHROMA_COLLECTION_NAME
cohere_api_key = os.getenv("COHERE_API_KEY")

cohere_client = cohere.Client(api_key=cohere_api_key)


def get_prompt_template() -> str:
    template = """
                Given the following extracted parts of a long document ("SOURCES") and a question ("QUESTION"), create a final answer max one paragraph long.
                Don't try to make up an answer and use the text in the SOURCES only for the answer. If you don't know the answer, just say that you don't know. 
                QUESTION: {question}
                =========
                SOURCES:
                {summaries}
                =========
                ANSWER:
               """
    return template


def get_summaries_for_query(query) -> str:
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L12-v2"
    )

    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection = chroma_client.get_collection(
        name=collection_name, embedding_function=sentence_transformer_ef
    )  # use the same embedding function that was used when docements were saved

    results = collection.query(query_texts=[f"{query}"], n_results=3)

    if results["documents"]:
        all_documents = ""
        for i, doc in enumerate(results["documents"]):
            doc_text = " ".join(doc)
            all_documents += doc_text + "\n\n"

        return all_documents
    else:
        return "No stored response found"


def chat(query, history):
    history_cohere_format = []
    for human, ai in history:
        history_cohere_format.append({human, ai})

    prompt = get_prompt_template()
    summaries = get_summaries_for_query(query)
    template = get_prompt_template()
    prompt = template.format(question=query, summaries=summaries)

    response = cohere_client.chat(
        message=prompt,
        temperature=0,  # chat_history=history_cohere_format
    )
    return response.text


chat_interface = gradio.ChatInterface(
    chat,
    chatbot=gradio.Chatbot(height=500),
    textbox=gradio.Textbox(placeholder="Ask me a question", container=True, scale=7),
    title="KB Bot",
    description="Ask questions about accounting, leadership, architecture guidelines and get response from the knowledge base",
    theme="default",
    examples=[
        "Tell me about swimlanes in architecture guidelines",
        "How to invoice accounting?",
        "Give some examples of actions from managers of one",
    ],
    cache_examples=True,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
)

chat_interface.launch()
