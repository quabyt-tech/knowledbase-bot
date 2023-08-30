import os
import argparse
from typing import List
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from langchain.schema import Document
from langchain.document_loaders import (
    UnstructuredMarkdownLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import constants

persist_dir = constants.CHROMA_PERSIST_DIR
collection_name = constants.CHROMA_COLLECTION_NAME


def chunk(docs: List[Document], chunk_size=1000, chunk_overlap=0) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)


## Load and split the files into Documents, may directly use DirectoryLoader
def load_documents_from_files(documents_directory: str) -> List[Document]:
    print("Loading docs...")
    documents = []
    files = os.listdir(documents_directory)
    for file in tqdm(files):
        file_path = os.path.join(documents_directory, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            pdf_docs = loader.load()
            chunks = chunk(pdf_docs)
            documents.extend(chunks)
        elif file.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(
                file_path
            )  # try UnstructuredWordDocumentLoader as well
            word_docs = loader.load()
            chunks = chunk(word_docs)
            documents.extend(chunks)
        elif file.endswith((".md")):
            loader = UnstructuredMarkdownLoader(file_path)
            md_docs = loader.load()
            chunks = chunk(md_docs)
            documents.extend(chunks)
    print(f"{len(documents)} documents loaded in memory")
    return documents


# Save the documents to the database, Chroma automatically generates embeddings based on embedding function
def save_documents(documents: List[Document]) -> None:
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L12-v2"  # Chroma uses all-MiniLM-L6-v2 by default. Another good choice: e5-small-v2
    )
    # Instantiate a persistent chroma client in the persist_directory.
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    collection = chroma_client.get_or_create_collection(
        name=collection_name, embedding_function=sentence_transformer_ef
    )

    # Create ids from the current count
    # count = collection.count() # uncomment if you want to allow duplicate docs, e.g. multiple runs with same docs
    # print(f"Collection already contains {count} documents")

    count = 0  # allows overwriting any existing docs with the same ids
    ids = [str(i) for i in range(count, count + len(documents))]
    metadatas = [documents[i].metadata for i in range(count, count + len(documents))]

    # Load the documents in batches of 100
    print()  # blank line
    for i in tqdm(
        range(0, len(documents), 100),
        desc="Saving documents to database...",
        unit_scale=100,
    ):
        # documents is an object from langchain library, just pull out the page_content strings to save
        page_contents = [doc.page_content for doc in documents[i : i + 100]]
        collection.add(
            ids=ids[i : i + 100],
            documents=page_contents,
            metadatas=metadatas[i : i + 100],
        )

    new_count = collection.count()
    print(f"Added {new_count - count} documents")


def main(
    documents_directory: str,
) -> None:
    documents = load_documents_from_files(documents_directory)

    save_documents(documents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load documents from a directory into a Chroma collection"
    )

    # Read the data directory, default value will be used if a parameter is not supplied
    parser.add_argument(
        "--data_dir",
        type=str,
        default="ORG-KB",
        help="The directory where data files are stored",
    )

    args = parser.parse_args()

    main(documents_directory=args.data_dir)
