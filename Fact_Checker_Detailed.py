"""
Overall Algorithm Outline:

1.  Setup & Data Loading:
    - Load Ramayana CSV, extract verses, and initialize Sentence Transformer for embeddings.
    - Load/compute and cache verse embeddings using the Sentence Transformer.
    - Build a FAISS index for efficient similarity search over verse embeddings.
    - Load a GGUF-formatted LLM (e.g., Mistral-7B-Instruct) for fact-checking.

2.  Verse Retrieval (Similarity Search):
    - For a given user statement, generate its embedding using the Sentence Transformer.
    - Query the FAISS index with the statement embedding to retrieve the top-K (e.g., 7) most semantically similar verses from the Ramayana, along with their locations (Kanda, Sarga, Shloka).

3.  LLM-based Fact-Checking:
    - Select a subset of retrieved verses (e.g., top 5) and format them with their locations.
    - Construct a prompt for the LLM, instructing it to act as a Ramayana scholar and determine if the user's statement is TRUE, FALSE, or UNDETERMINED based *only* on the provided verses, and to give an explanation.
    - Send the prompt to the GGUF LLM and get its response.

4.  Response Parsing & Reference Heuristic:
    - Parse the LLM's raw response to extract the decision (TRUE/FALSE/UNDETERMINED) and the explanation.
    - If the decision is TRUE, apply a heuristic to select a "best" reference: iterate through verses sent to the LLM, preferring those from earlier Kandas (defined by KANDA_ORDER) and with some keyword overlap with the user statement.
    - For FALSE/UNDETERMINED, the top FAISS-retrieved verse is typically used as the reference.

5.  Logging & Output:
    - Log the user statement, LLM's decision, explanation, selected reference verse text & location, and raw LLM response to a CSV file.
    - Display results to the user, supporting both interactive and batch (from input CSV file) processing.
"""

# --- Import necessary libraries ---
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import faiss
import numpy as np
import os
import sys
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import csv

# --- Configuration ---
# All parameters and file paths are centralized here for easy management.
CONFIG = {
    # --- Hugging Face Repository Details ---
    "hf_repo_id": "dishasahu/ramayana-fact-checker-assets",

    # --- Filenames (must match the names in the HF repo) ---
    "ramayana_dataset_filename": "Valmiki_Ramayana_Dataset.csv",
    "embeddings_cache_filename": "ramayana_verse_embeddings_mpnet.npy",
    "gguf_model_filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",

    # --- Model and System Settings ---
    "embedding_model_name": 'sentence-transformers/all-mpnet-base-v2',
    "relevance_score_threshold": 1.1,  # L2 distance; higher means less similar.

    # --- CSV Column Names ---
    "verse_column_name": 'English Translation',
    "kanda_column": 'Kanda/Book',
    "sarga_column": 'Sarga/Chapter',
    "shloka_column": 'Shloka/Verse',

    # --- Output file ---
    "output_csv_path": "prediction_output.csv",

    # --- Heuristics ---
    "kanda_order": [
        "Bala Kanda", "Baala Kanda", "Ayodhya Kanda", "Aranya Kanda",
        "Kishkindha Kanda", "Kishkinda Kanda", "Sundara Kanda", "Yuddha Kanda"
    ],
    "stop_words": {
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
        'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'did', 'do',
        'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have',
        'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
        'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not',
        'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 's',
        'same', 'she', 'should', 'so', 'some', 'such', 't', 'than', 'that', 'the', 'their', 'theirs', 'them',
        'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under',
        'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
        'why', 'will', 'with', 'you', 'your', 'yours', 'yourself', 'yourselves'
    }
}

# --- Global Variables ---
df = None
valid_df = None
verses = None
embedder = None
verse_embeddings_np = None
index = None
llm_model = None

def download_from_hf_hub(repo_id, filename):
    """Downloads a file from a Hugging Face Hub repo if it doesn't exist locally."""
    if os.path.exists(filename):
        print(f"File '{filename}' already exists. Skipping download.")
        return
    
    print(f"Downloading '{filename}' from repo '{repo_id}'...")
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir='.', # Download to the current directory
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        print(f"FATAL: Could not download '{filename}' from repo '{repo_id}'. Error: {e}")
        sys.exit(1)

def setup_models_and_data():
    """
    Handles all one-time setup: downloading data/models from Hugging Face, loading
    them into memory, and building the FAISS index.
    """
    global df, valid_df, verses, embedder, verse_embeddings_np, index, llm_model
    print("--- 1. Setting up models and data ---")

    # --- Download all required files from your Hugging Face repo ---
    repo_id = CONFIG['hf_repo_id']
    download_from_hf_hub(repo_id, CONFIG['ramayana_dataset_filename'])
    download_from_hf_hub(repo_id, CONFIG['embeddings_cache_filename'])
    download_from_hf_hub(repo_id, CONFIG['gguf_model_filename'])

    # --- Load Ramayana dataset ---
    print("\nLoading dataset and models...")
    try:
        df = pd.read_csv(CONFIG['ramayana_dataset_filename'])
    except FileNotFoundError:
        print(f"FATAL: CSV file not found at {CONFIG['ramayana_dataset_filename']}. Exiting."); sys.exit(1)

    required_cols = [CONFIG['verse_column_name'], CONFIG['kanda_column'], CONFIG['sarga_column'], CONFIG['shloka_column']]
    if not all(col in df.columns for col in required_cols):
        print(f"FATAL: Input CSV must contain columns: {required_cols}. Exiting."); sys.exit(1)

    valid_df = df[df[CONFIG['verse_column_name']].notna()].copy()
    verses = valid_df[CONFIG['verse_column_name']].astype(str).tolist()
    print(f"Loaded {len(verses)} verses from dataset.")

    # --- Load SBERT and Embeddings ---
    embedder = SentenceTransformer(CONFIG['embedding_model_name'])
    
    verse_embeddings_np = np.load(CONFIG['embeddings_cache_filename'])
    if verse_embeddings_np.shape[0] != len(verses):
        print("FATAL: Embeddings file does not match the number of verses. Please regenerate and re-upload.")
        sys.exit(1)

    # --- Build FAISS Index ---
    dim = verse_embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(verse_embeddings_np)
    print("FAISS index built.")

    # --- Load GGUF LLM ---
    llm_model = Llama(model_path=CONFIG['gguf_model_filename'], n_ctx=2048, n_gpu_layers=-1, verbose=False)
    print("GGUF LLM loaded successfully.")
    print("--- Setup Complete ---")

def retrieve_top_verses(query: str, k_retrieval: int = 7) -> list:
    """Uses FAISS to find the top k most similar verses for a given query."""
    q_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_vec, k_retrieval)
    results = []
    for i in range(len(indices[0])):
        faiss_idx = indices[0][i]
        meta = valid_df.iloc[faiss_idx]
        loc = f"{meta[CONFIG['kanda_column']]}, Sarga {meta[CONFIG['sarga_column']]}, Shloka {meta[CONFIG['shloka_column']]}"
        results.append({"text": verses[faiss_idx], "location": loc, "score": float(distances[0][i]), "meta_row": meta})
    return results

def get_kanda_name_from_location_string(location_str: str) -> str | None:
    """Helper to extract Kanda name from 'Kanda, Sarga, Shloka' string."""
    try:
        return location_str.split(',')[0].strip()
    except IndexError:
        return None

def format_output_row(prediction: str, top_verse_info: dict | None) -> dict:
    """Helper to structure the output dictionary with prediction and reference."""
    if top_verse_info:
        meta = top_verse_info['meta_row']
        return {
            "Prediction": prediction,
            "Reference_Kanda": meta[CONFIG['kanda_column']],
            "Reference_Sarga": meta[CONFIG['sarga_column']],
            "Reference_Shloka": meta[CONFIG['shloka_column']],
            "Reference_Verse": top_verse_info['text']
        }
    else:
        return {
            "Prediction": prediction,
            "Reference_Kanda": "N/A",
            "Reference_Sarga": "N/A",
            "Reference_Shloka": "N/A",
            "Reference_Verse": "N/A"
        }

def get_prediction_for_statement(user_statement: str) -> dict:
    """
    Core fact-checking function. Returns a dictionary with prediction and reference.
    """
    # Handle empty or invalid statements upfront.
    if not isinstance(user_statement, str) or len(user_statement.strip()) < 10:
        return format_output_row("NOT RELEVANT", None)

    # Step 1: Retrieve verses and find the most relevant one.
    all_retrieved_verses = retrieve_top_verses(user_statement, k_retrieval=7)
    top_verse = all_retrieved_verses[0] if all_retrieved_verses else None

    # If no verse is found or the best match is still not similar enough, the statement is irrelevant.
    if not top_verse or top_verse['score'] > CONFIG['relevance_score_threshold']:
        return format_output_row("NOT RELEVANT", top_verse)

    # Step 2: Prepare context for the LLM.
    verses_for_llm = all_retrieved_verses[:5]
    verses_section = "Relevant Ramayana Verses:\n"
    for i, v_info in enumerate(verses_for_llm):
        verses_section += f"{i+1}. ({v_info['location']}): \"{v_info['text']}\"\n"
    
    # Step 3: Construct the precise prompt for the LLM.
    prompt_intro = "You are an expert scholar of the Valmiki Ramayana."
    prompt_task = ("Analyze the 'User Statement' based *only* on the 'Relevant Ramayana Verses' provided. "
                   "Your answer MUST start with one of three words: TRUE, FALSE, or UNDETERMINED. "
                   "After the word, provide a brief explanation for your decision.")
    full_prompt = f"[INST] {prompt_intro}\n{prompt_task}\n\n{verses_section}\nUser Statement: \"{user_statement}\"\n\nAnswer: [/INST]"

    # Step 4: Query the LLM.
    try:
        output = llm_model(full_prompt, max_tokens=150, stop=["</s>", "[/INST]"], temperature=0.1, echo=False)
        llm_response = output['choices'][0]['text'].strip()
    except Exception as e:
        print(f"Warning: LLM generation error for statement '{user_statement[:50]}...': {e}")
        # Treat LLM errors as an inability to determine, but still provide the top reference.
        return format_output_row("NOT RELEVANT", top_verse)

    # Step 5: Parse the LLM response robustly.
    prediction = "NOT RELEVANT" # Default prediction
    match = re.search(r"\b(TRUE|FALSE|UNDETERMINED)\b", llm_response, re.IGNORECASE)
    if match:
        decision_str = match.group(1).upper()
        if decision_str == "TRUE":
            prediction = "TRUE"
        elif decision_str == "FALSE":
            prediction = "FALSE"
        # "UNDETERMINED" maps to "NOT RELEVANT"
    
    return format_output_row(prediction, top_verse)

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Validate Input ---
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(__file__)} <path_to_input_csv>")
        sys.exit(1)
    
    input_csv_path = sys.argv[1]
    if not os.path.exists(input_csv_path):
        print(f"FATAL: Input file not found at '{input_csv_path}'")
        sys.exit(1)

    # --- Run Setup ---
    setup_models_and_data()

    # --- Process Input Statements ---
    print(f"\n--- 2. Processing statements from '{input_csv_path}' ---")
    try:
        input_df = pd.read_csv(input_csv_path)
        if "Statement" not in input_df.columns:
            print(f"FATAL: Input CSV must have a 'Statement' column.")
            sys.exit(1)
    except Exception as e:
        print(f"FATAL: Could not read input CSV. Error: {e}")
        sys.exit(1)

    statements_to_process = input_df["Statement"].tolist()
    
    # --- Open output file and write results simultaneously ---
    output_csv_path = CONFIG['output_csv_path']
    # Define the headers for the output file
    fieldnames = ["Statement", "Prediction", "Reference_Kanda", "Reference_Sarga", "Reference_Shloka", "Reference_Verse"]
    
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for statement in tqdm(statements_to_process, desc="Fact-checking statements"):
                # Get the dictionary containing prediction and reference
                prediction_data = get_prediction_for_statement(statement)
                
                # Combine the original statement with the prediction results
                row_to_write = {"Statement": statement}
                row_to_write.update(prediction_data)
                
                # Write the complete row to the CSV file
                writer.writerow(row_to_write)

    except Exception as e:
        print(f"FATAL: Could not write to output file '{output_csv_path}'. Error: {e}")
        sys.exit(1)


    print(f"\n--- 3. Processing Complete ---")
    print(f"Output successfully saved to '{CONFIG['output_csv_path']}'")

    # --- Cleanup ---
    if llm_model is not None:
        print("Freeing LLM resources...")
        del llm_model
    
    print("Done.")
