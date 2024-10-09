import os
import fitz
import re
import pandas as pd
import numpy as np
import torch
import textwrap
from time import perf_counter as timer
from spacy.lang.en import English
from sentence_transformers import util, SentenceTransformer
from dotenv import load_dotenv
from groq import Groq
import streamlit as st
import requests
import warnings
warnings.filterwarnings('ignore')

# Set device for PyTorch (either 'cuda' or 'cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define default file paths and models
embeddings_save_path = 'text_and_embeddings.csv'
model_name = 'all-MiniLM-L6-v2'
model_llm_name = 'llama3-8b-8192'
demo_embeddings_url = 'https://huggingface.co/datasets/InvictusRudra/rag_embeddings/resolve/main/text_and_embeddings.csv'

def print_wrapped(text, wrap_length=80):
    """Wrap text to fit within a specific width."""
    wrapped_text = textwrap.fill(text, wrap_length)
    st.write(wrapped_text)

def load_demo_embeddings():
    """Load pre-generated demo embeddings from Hugging Face."""
    st.write('Loading demo embeddings from Hugging Face...')
    response = requests.get(demo_embeddings_url)
    with open('demo_embeddings.csv', 'wb') as f:
        f.write(response.content)
    
    text_and_embeddings = pd.read_csv('demo_embeddings.csv')
    text_and_embeddings["embedding"] = text_and_embeddings["embedding"].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=" "))
    embeddings = torch.tensor(np.stack(text_and_embeddings["embedding"].to_list(), axis=0, dtype=np.float32)).to(device)
    pages_chunks_mod = text_and_embeddings.to_dict(orient="records")
    st.write("Demo embeddings loaded successfully.")
    return embeddings, pages_chunks_mod

def preprocess(pdf_file):
    """Preprocess the PDF file to extract text, chunk it, and create embeddings."""
    # Process the PDF file
    def formatter(text):
        clean = text.replace("\n", " ").strip()
        return clean

    def open_read(pdf_file):
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        pages = []
        total_pages = doc.page_count
        progress_bar = st.progress(0)  # Initialize progress bar
        for page_no, page in enumerate(doc):
            text = formatter(page.get_text())
            pages.append({"page_no": page_no,
                          "page_char_count": len(text),
                          "page_word_count": len(text.split(" ")),
                          "page_sentence_count": len(text.split(". ")),
                          "page_token_count": len(text)/4,
                          "text": text})
            progress_bar.progress((page_no + 1) / total_pages)
        return pages

    st.write(f'Processing PDF . . .')
    pages = open_read(pdf_file)

    # Sentence splitting
    st.write(f'Sentencizing . . .')
    nlp = English()
    nlp.add_pipe("sentencizer")
    progress_bar = st.progress(0)
    for i, item in enumerate(pages):
        item["sentences"] = list(nlp(item["text"]).sents)
        item["sentences"] = [str(x) for x in item["sentences"]]
        item["sentence_count_spacy"] = len(item["sentences"])
        progress_bar.progress((i + 1) / len(pages))
    st.write(f'Finished sentencizing \n')

    # Chunking text
    st.write(f'Chunking . . .')
    chunk_size = 10

    def split_list(input_list: list[str], slice_size: int = chunk_size) -> list[list[str]]:
        return [input_list[i: i+slice_size] for i in range(0, len(input_list), slice_size)]

    progress_bar = st.progress(0)
    for i, item in enumerate(pages):
        item["sentence_chunks"] = split_list(item["sentences"], chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])
        progress_bar.progress((i + 1) / len(pages))

    pages_chunks = []
    for item in pages:
        for chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_no"]
            joined_chunk = "".join(chunk).replace("  ", " ").strip()
            joined_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_chunk)  # ".A" -> ". A"
            chunk_dict["sentence_chunk"] = joined_chunk
            chunk_dict["chunk_char_count"] = len(joined_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_chunk) / 4
            pages_chunks.append(chunk_dict)

    df = pd.DataFrame(pages_chunks)
    min_token_length = 30
    pages_chunks_mod = df[df["chunk_token_count"] >= min_token_length].to_dict(orient="records")
    st.write(f'Finished chunking \n')

    # Embedding creation
    st.write(f'Beginning embedding . . .')
    start_time = timer()
    embed_model = SentenceTransformer(model_name_or_path=model_name, device=device)
    progress_bar = st.progress(0)
    for i, item in enumerate(pages_chunks_mod):
        item["embedding"] = embed_model.encode(item["sentence_chunk"])
        progress_bar.progress((i + 1) / len(pages_chunks_mod))

    st.write(f'Saving embeddings to {embeddings_save_path}')
    text_and_embeddings = pd.DataFrame(pages_chunks_mod)
    text_and_embeddings.to_csv(embeddings_save_path, index=False)
    end_time = timer()
    st.write(f'Finished embedding \n')
    st.write(f'Time taken to embed: {end_time - start_time:.4f} seconds')


def retrieve(query, embeddings, model, num_returns=5):
    """Retrieve top results based on the query and embeddings."""
    query_embedding = model.encode(query, convert_to_tensor=True).to(device=device)
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    scores, indices = torch.topk(dot_scores, k=num_returns)
    return scores, indices


def prompt_formatter(query: str, context_items: list[dict]) -> str:
    """Format the query and context into a prompt for the LLM."""
    context = '- ' + "\n- ".join([item["sentence_chunk"] for item in context_items])
    prompt = f"""Based on the following context items, answer the query. 
    Give yourself room to think by extracting relevant passages from the context before answering the query.
    Context_items: {context}
    Query: {query}
    Return your answer in this format:
    Query: <the query here>
    Answer: <Your answer>
    """
    return prompt


def ask(query, embeddings=None, pages_chunks_mod=None):
    """Perform the complete Retrieval-Augmented Generation (RAG) process."""
    st.write('Loading embeddings...')
    if embeddings is None:
        text_and_embeddings = pd.read_csv(embeddings_save_path)
        text_and_embeddings["embedding"] = text_and_embeddings["embedding"].apply(
            lambda x: np.fromstring(x.strip("[]"), sep=" "))
        embeddings = torch.tensor(np.stack(text_and_embeddings["embedding"].to_list(), axis=0, dtype=np.float32)).to(device)
        pages_chunks_mod = text_and_embeddings.to_dict(orient="records")

    embed_model = SentenceTransformer(model_name_or_path=model_name, device=device)
    load_dotenv()
    client = Groq(
        api_key=os.getenv("GROQ_API_KEY"),
    )

    _, indices = retrieve(query, embeddings, embed_model)
    context_items = [pages_chunks_mod[i] for i in indices]
    prompt = prompt_formatter(query, context_items=context_items)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_llm_name,
    )
    rag_answer = chat_completion.choices[0].message.content
    return rag_answer.replace(prompt, '')


def main():
    """Main function to run the Streamlit app."""
    st.title("RAG Model with Groq API and Streamlit")

    # Upload section for PDF files
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    # Checkbox for recreating embeddings
    recreate_embeddings = st.checkbox("Recreate embeddings", value=False)

    # Button to use demo embeddings
    use_demo = st.checkbox("Use demo embeddings on nutrition textbook", value=False)

    # Enter query
    query = st.text_input("Enter your query:")

    # Button to run the process
    if st.button("Run"):
        if use_demo:
            if query:
                # Load demo embeddings and run the ask function
                embeddings, pages_chunks_mod = load_demo_embeddings()
                with st.spinner("Retrieving and generating answer..."):
                    result = ask(query, embeddings, pages_chunks_mod)
                    st.success("Answer generated!")
                    st.subheader("Answer:")
                    st.write(result)
        elif uploaded_file:
            # Preprocessing and embedding creation
            if recreate_embeddings or not os.path.exists(embeddings_save_path):
                st.write("Processing uploaded file and creating embeddings...")
                preprocess(uploaded_file)

            if query:
                # Answer generation and retrieval
                with st.spinner("Retrieving and generating answer..."):
                    result = ask(query)
                    st.success("Answer generated!")
                    st.subheader("Answer:")
                    st.write(result)
        else:
            st.warning("Please upload a PDF file before running, or use demo embeddings.")


if __name__ == '__main__':
    main()
