from langchain_community.document_loaders import PyPDFLoader
import os
import datetime


def load_all():
    pdf_folder_path = './data/docs'
    doc_list = [s for s in os.listdir(pdf_folder_path) if s.endswith('.pdf')]
    num_of_docs = len(doc_list)

    general_start = datetime.datetime.now()
    print("starting the loop...")
    loop_start = datetime.datetime.now()

    print("Main Vector database created. Start iteration and merging...")
    for i in range(0, num_of_docs):
        print(doc_list[i])
        print(f"loop position {i}")
        loader = PyPDFLoader(os.path.join(pdf_folder_path, doc_list[i]))
        start = datetime.datetime.now()
        doc_parts = loader.load()

        char_count = 0
        for doc_part in doc_parts:
            char_count += len(doc_part.page_content)
        print(char_count)

        end = datetime.datetime.now()
        elapsed = end - start

        print(f"completed in {elapsed}")
        print("-----------------------------------")
    loop_end = datetime.datetime.now()
    loop_elapsed = loop_end - loop_start
    print(f"All documents processed in {loop_elapsed}")
    general_end = datetime.datetime.now()
    general_elapsed = general_end - general_start
    print(f"All indexing completed in {general_elapsed}")
    print("-----------------------------------")


def load_single_pdf(pdf_path):
    """Loads a single PDF and returns its parts."""
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return []
    try:
        loader = PyPDFLoader(pdf_path)
        doc_parts = loader.load()
        return doc_parts
    except Exception as e:
        print(f"Error loading PDF {pdf_path}: {e}")
        return []
