import os 
import requests
import fitz
from tqdm.auto import tqdm
from spacy.lang.en import English
import numpy as np
import pandas as pd
import re
from sentence_transformers import util, SentenceTransformer
import random
import torch
from time import perf_counter as timer
import textwrap
from dotenv import load_dotenv
from groq import Groq

import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
embeddings_save_path = 'text_and_embeddings.csv'
# pdf_path = './human-nutrition-text.pdf'
pdf_path = 'PG_Information_Brochure_Jul_2021.pdf'
# model_name = 'all-mpnet-base-v2'
model_name = 'all-MiniLM-L6-v2'
model_llm_name = 'llama3-8b-8192'

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

def preprocess():
    pdf_url = 'https://www.iitk.ac.in/doaa/DOAA/PG_Information_Brochure_Jul_2021.pdf' # enter url here
    if not os.path.exists(pdf_path):
        print(f'Downloading file . . .')
        response = requests.get(pdf_url)
        if response.status_code == 200:
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            print(f'File is downloaded as {pdf_path}')
        else:
            print(f'[ERROR] {response.status_code}')
    else:
        print(f'File {pdf_path} already exists')

    def formatter(text):
        clean = text.replace("\n", " ").strip()
        return clean

    def open_read(pdf_path):
        doc = fitz.open(pdf_path)
        pages = []
        for page_no, page in tqdm(enumerate(doc)):
            text = formatter(page.get_text())
            pages.append({"page_no" : page_no-41 , 
                        "page_char_count" : len(text), 
                        "page_word_count" : len(text.split(" ")),
                        "page_sentence_count" : len(text.split(". ")), 
                        "page_token_count" : len(text)/4, 
                        "text": text})
        return pages

    print(f'Processing PDF {pdf_path} . . .')
    pages = open_read(pdf_path=pdf_path)

    print(f'Sentencizing . . .')
    nlp = English()
    nlp.add_pipe("sentencizer")
    for item in tqdm(pages):
        item["sentences"] = list(nlp(item["text"]).sents)
        item["sentences"] = [str(x) for x in item["sentences"]]
        item["sentence_count_spacy"] = len(item["sentences"])
    print(f'Finished sentencizing \n')

    print(f'Chunking . . .')
    chunk_size = 10
    def split_list(input_list: list[str], 
                slice_size: int = chunk_size) -> list[list[str]]:
        return [input_list[i : i+slice_size] for i in range(0, len(input_list), slice_size)]

    for item in tqdm(pages):
        item["sentence_chunks"] = split_list(item["sentences"], chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])

    pages_chunks = []
    for item in tqdm(pages):
        for chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_no"]
            joined_chunk = "".join(chunk).replace("  ", " ").strip()
            joined_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_chunk) # ".A" -> ". A"
            chunk_dict["sentence_chunk"] = joined_chunk
            chunk_dict["chunk_char_count"] = len(joined_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_chunk) / 4
            pages_chunks.append(chunk_dict)

    df = pd.DataFrame(pages_chunks)
    min_token_length = 30
    pages_chunks_mod = df[df["chunk_token_count"] >= min_token_length].to_dict(orient="records")
    print(f'Finished chunking \n')

    print(f'Beginning embedding . . .')
    start_time = timer()
    embed_model = SentenceTransformer(model_name_or_path=model_name, device=device)
    for item in tqdm(pages_chunks_mod): 
        item["embedding"] = embed_model.encode(item["sentence_chunk"])

    print(f'Saving embeddings to {embeddings_save_path}')
    text_and_embeddings = pd.DataFrame(pages_chunks_mod)
    text_and_embeddings.to_csv(embeddings_save_path, index=False)
    end_time = timer()
    print(f'Finished embedding \n')
    print(f'Time taken to embed : {end_time-start_time:.4f}')

def retrieve(query, embeddings, model=model_name, num_returns=5, print_time=False, print_results=False, data=None):

    query_embedding = model.encode(query, convert_to_tensor=True).to(device=device)
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    end_time = timer()
    scores, indices = torch.topk(dot_scores, k=num_returns)

    if print_time:
        print(f'Time to retrieve : {end_time-start_time:.6f}')
    
    if print_results:
        assert data 
        print(f'Query: {query}')
        for score, idx in zip(scores, indices):  
            print(f'Score : {score}')
            print(f'Text: \n')
            print_wrapped(data[idx]["sentence_chunk"])
            print(f'Page number: {data[idx]["page_number"]}')
            print('\n')
        
    return scores, indices

def prompt_formatter(query:str, context_items:list[dict]) -> str:
    
    context = '- ' + "\n- ".join([item["sentence_chunk"] for item in context_items])
    prompt = f"""Based on the following context items answer the query. 
    Give yourself room to think by extracting relevant passages from the context before answering the query.
    Don't return the thinking, return only the answer.
    Make sure your answers are as explanatory as possible.
    Context_items: {context}
    Query : {query}
    Return your answers in the below format:
    Query : <the query here>
    Answer : <Your answer>
    """
    return prompt

def ask(query, print_sources:bool=False):
    start_time = timer()
    text_and_embeddings = pd.read_csv(embeddings_save_path)
    text_and_embeddings["embedding"] = text_and_embeddings["embedding"].apply(lambda x : np.fromstring(x.strip("[]"), sep=" "))
    embeddings = torch.tensor(np.stack(text_and_embeddings["embedding"].to_list(), axis=0, dtype=np.float32)).to(device)
    pages_chunks_mod = text_and_embeddings.to_dict(orient="records")
    print(f'Embedding dimension : {embeddings.shape[1]}')

    embed_model = SentenceTransformer(model_name_or_path=model_name, device=device)
    load_dotenv()
    client = Groq(
        api_key=os.getenv("GROQ_API_KEY"),
    )
    _, indices = retrieve(query, embeddings, embed_model, print_time=True, data=pages_chunks_mod)
    print(f'Retrieved relevant sources \n')
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
    end_time = timer()
    print(f'Time taken to answer : {end_time-start_time:.4f}')
    if print_sources:
        print(f'Relevant sources are : \n')
        for text in context_items:
            print(f'- {text["sentence_chunk"]}')
        print('\n\n')
    return rag_answer.replace(prompt, '')

    

if __name__ == '__main__':
    if not os.path.exists(embeddings_save_path):
        preprocess()
        print(f'Preprocessing done \n')
    query = input('Enter your query now : ')
    result = ask(query=query, print_sources=True)
    print(result)



