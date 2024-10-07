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
import gradio as gr
import warnings

warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
embeddings_save_path = 'text_and_embeddings.csv'
model_name = 'all-MiniLM-L6-v2'
model_llm_name = 'llama3-8b-8192'

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    return wrapped_text

def preprocess(pdf_file):
    def formatter(text):
        clean = text.replace("\n", " ").strip()
        return clean

    def open_read(pdf_file):
        doc = fitz.open(pdf_file)
        pages = []
        total_pages = doc.page_count
        for page_no, page in enumerate(doc):
            text = formatter(page.get_text())
            pages.append({"page_no": page_no,
                          "page_char_count": len(text),
                          "page_word_count": len(text.split(" ")),
                          "page_sentence_count": len(text.split(". ")),
                          "page_token_count": len(text)/4,
                          "text": text})
        return pages

    pages = open_read(pdf_file)

    nlp = English()
    nlp.add_pipe("sentencizer")
    for i, item in enumerate(pages):
        item["sentences"] = list(nlp(item["text"]).sents)
        item["sentences"] = [str(x) for x in item["sentences"]]
        item["sentence_count_spacy"] = len(item["sentences"])

    chunk_size = 10

    def split_list(input_list: list[str], slice_size: int = chunk_size) -> list[list[str]]:
        return [input_list[i: i+slice_size] for i in range(0, len(input_list), slice_size)]

    for i, item in enumerate(pages):
        item["sentence_chunks"] = split_list(item["sentences"], chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])

    pages_chunks = []
    for item in pages:
        for chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_no"]
            joined_chunk = "".join(chunk).replace("  ", " ").strip()
            joined_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_chunk)
            chunk_dict["sentence_chunk"] = joined_chunk
            chunk_dict["chunk_char_count"] = len(joined_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_chunk) / 4
            pages_chunks.append(chunk_dict)

    df = pd.DataFrame(pages_chunks)
    min_token_length = 30
    pages_chunks_mod = df[df["chunk_token_count"] >= min_token_length].to_dict(orient="records")

    start_time = timer()
    embed_model = SentenceTransformer(model_name_or_path=model_name, device=device)
    for i, item in enumerate(pages_chunks_mod):
        item["embedding"] = embed_model.encode(item["sentence_chunk"])

    text_and_embeddings = pd.DataFrame(pages_chunks_mod)
    text_and_embeddings.to_csv(embeddings_save_path, index=False)
    end_time = timer()

def retrieve(query, embeddings, model, num_returns=5):
    query_embedding = model.encode(query, convert_to_tensor=True).to(device=device)
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    scores, indices = torch.topk(dot_scores, k=num_returns)
    return scores, indices

def prompt_formatter(query: str, context_items: list[dict]) -> str:
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

def ask(query):
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

def main(pdf_file, query, recreate_embeddings):
    if pdf_file:
        if recreate_embeddings or not os.path.exists(embeddings_save_path):
            preprocess(pdf_file)

        if query:
            result = ask(query)
            return result
    return "Please upload a PDF file and enter a query."

demo = gr.Interface(
    fn=main,
    inputs=[
        gr.File(label="Upload a PDF file"), 
        gr.Textbox(label="Enter your query"), 
        gr.Checkbox(label="Recreate embeddings", value=False)
    ],
    outputs="text",
    title="Simple Rag pipeline"
)

if __name__ == '__main__':
    demo.launch(share=True)
