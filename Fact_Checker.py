# This script performs fact-checking on statements related to the Valmiki Ramayana.
# It uses a Retrieval-Augmented Generation (RAG) pipeline.
# 1. It downloads the necessary dataset, embeddings, and a Mistral-7B GGUF model
#    from the 'dishasahu/ramayana-fact-checker-assets' Hugging Face repository.
# 2. For each input statement, it retrieves the most relevant verses from the Ramayana
#    using a FAISS vector index.
# 3. It uses the retrieved verses as context for a Large Language Model (LLM) to
#    determine if the statement is TRUE, FALSE, or NOT RELEVANT.
# 4. The final predictions are saved to 'prediction_output.csv'.
#
# How to Run:
# 1. Make sure you have an input CSV file (e.g., 'test_statements.csv') in the same
#    directory. This file must have a column named "Statement".
# 2. Run the script from your terminal:
#    python Fact_Checker.py test_statements.csv

# Import necessary libraries
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import faiss
import numpy as np
import os

# --- Configuration & Model Loading ---

# Path to the source dataset CSV file containing the Ramayana verses.
CSV_PATH = 'Valmiki_Ramayana_Dataset.csv'
# Column name in the CSV that holds the English translation of the verses.
VERSE_COLUMN_NAME = 'English Translation'
# Column names for the location metadata of each verse.
KANDA_COLUMN = 'Kanda/Book'
SARGA_COLUMN = 'Sarga/Chapter'
SHLOKA_COLUMN = 'Shloka/Verse'
# Name of the pre-trained Sentence Transformer model for generating embeddings.
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
# Path to the local GGUF format Large Language Model file.
GGUF_MODEL_PATH = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
# Path to save/load the pre-computed verse embeddings to speed up subsequent runs.
EMBEDDINGS_CACHE_PATH = "ramayana_verse_embeddings_mpnet.npy"
# Path for the output CSV file where results will be logged.
LOG_CSV_PATH = "ramayana_fact_checker_output.csv"
# Defines the canonical order of the Ramayana books (Kandas) for chronological sorting.
KANDA_ORDER = [
    "Bala Kanda", "Baala Kanda", "Ayodhya Kanda", "Aranya Kanda",
    "Kishkindha Kanda", "Kishkinda Kanda", "Sundara Kanda", "Yuddha Kanda"
]

# --- Global Variables ---
# These variables will be initialized in setup_models_and_data() and used throughout the script.
df = None                   # The main pandas DataFrame holding the dataset.
valid_df = None             # A filtered DataFrame containing only rows with valid verse text.
verses = None               # A list of all verse strings.
embedder = None             # The SentenceTransformer model instance.
verse_embeddings_np = None  # A numpy array of all verse embeddings.
index = None                # The FAISS index for fast similarity search.
llm_model = None            # The Llama.cpp model instance.

def setup_models_and_data():
    """
    Loads the dataset, initializes the embedding and LLM models, and builds the FAISS index.
    This function handles all one-time setup tasks.
    """
    # Make global variables modifiable within this function.
    global df, valid_df, verses, embedder, verse_embeddings_np, index, llm_model
    print("--- Setting up models and data ---")
    
    # Load the Ramayana dataset from the specified CSV file.
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"Loaded CSV '{CSV_PATH}'.")
    except FileNotFoundError:
        print(f"ERROR: CSV file not found at {CSV_PATH}. Exiting."); exit(1)

    # Verify that all required columns are present in the DataFrame.
    required_cols = [VERSE_COLUMN_NAME, KANDA_COLUMN, SARGA_COLUMN, SHLOKA_COLUMN]
    for col in required_cols:
        if col not in df.columns:
            print(f"ERROR: Column '{col}' not in CSV. Exiting."); exit(1)

    # Filter out rows where the verse translation is missing and create a list of verses.
    valid_df = df[df[VERSE_COLUMN_NAME].notna()].copy()
    verses = valid_df[VERSE_COLUMN_NAME].astype(str).tolist()
    print(f"Loaded {len(verses)} verses.")
    
    # Load the SentenceTransformer model for creating vector embeddings.
    print(f"Loading SBERT model ({EMBEDDING_MODEL_NAME})...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Check for a pre-computed cache of embeddings to save time.
    if os.path.exists(EMBEDDINGS_CACHE_PATH):
        try:
            verse_embeddings_np = np.load(EMBEDDINGS_CACHE_PATH)
            # Ensure the cached embeddings match the number of verses in the current dataset.
            if verse_embeddings_np.shape[0] == len(verses):
                print("Cached embeddings loaded.")
            else:
                # If there's a mismatch, discard the cache and recompute.
                print("Cache mismatch. Recomputing."); verse_embeddings_np = None
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing."); verse_embeddings_np = None

    # If embeddings weren't loaded from cache, compute them now.
    if verse_embeddings_np is None:
        print("Computing verse embeddings...")
        verse_embeddings_np = embedder.encode(verses, convert_to_numpy=True, show_progress_bar=True)
        # Save the newly computed embeddings to the cache file.
        np.save(EMBEDDINGS_CACHE_PATH, verse_embeddings_np)
        print(f"Embeddings computed & saved to {EMBEDDINGS_CACHE_PATH}.")

    # Build the FAISS index for efficient vector similarity search.
    dim = verse_embeddings_np.shape[1]  # Get the dimension of the embeddings.
    index = faiss.IndexFlatL2(dim)      # Use a simple L2 distance index.
    index.add(verse_embeddings_np)      # Add all verse embeddings to the index.
    print("FAISS index built.")

    # Load the GGUF Large Language Model using llama-cpp-python.
    if not os.path.exists(GGUF_MODEL_PATH):
        print(f"ERROR: GGUF file '{GGUF_MODEL_PATH}' not found. Exiting."); exit(1)
    print(f"Loading GGUF model from ({GGUF_MODEL_PATH})...")
    try:
        # Initialize the Llama model, offloading all layers to the GPU if possible.
        llm_model = Llama(model_path=GGUF_MODEL_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=False)
        print("GGUF LLM loaded.")
    except Exception as e:
        print(f"ERROR loading GGUF LLM: {e}. Exiting."); exit(1)
    print("--- Setup complete ---")

def retrieve_top_verses(query: str, k_retrieval: int = 7) -> list:
    """
    Encodes a query string and uses the FAISS index to find the top k most similar verses.
    """
    # Convert the user's query into a vector embedding.
    q_vec = embedder.encode([query], convert_to_numpy=True)
    # Search the FAISS index for the k nearest neighbors.
    distances, indices = index.search(q_vec, k_retrieval)
    results = []
    # Compile the results into a list of dictionaries.
    for i in range(len(indices[0])):
        faiss_idx = indices[0][i]
        meta = valid_df.iloc[faiss_idx] # Get metadata from the original DataFrame.
        loc = f"{meta[KANDA_COLUMN]}, Sarga {meta[SARGA_COLUMN]}, Shloka {meta[SHLOKA_COLUMN]}"
        results.append({"text": verses[faiss_idx], "location": loc, "score": float(distances[0][i]), "meta_row": meta})
    return results

def get_kanda_name_from_location_string(location_str: str) -> str | None:
    """
    A helper function to extract the Kanda (Book) name from a formatted location string.
    Example: "Bala Kanda, Sarga 1, Shloka 1" -> "Bala Kanda"
    """
    try:
        return location_str.split(',')[0].strip()
    except IndexError:
        return None

def is_ramayana_statement_correct(user_statement: str) -> dict:
    """
    The core fact-checking function. It retrieves relevant verses, prompts an LLM
    for an analysis, and parses the result.
    """
    # Define a default dictionary to return in case of errors.
    default_return = {
        "result_bool": None, "llm_raw_response": "N/A", "explanation": "N/A",
        "top_retrieved_verse_text": "N/A", "retrieved_reference": "N/A"
    }
    # Basic validation for the LLM and user input.
    if llm_model is None or not user_statement or len(user_statement.strip()) < 10:
        default_return["explanation"] = "LLM not ready or statement too short."; return default_return

    print(f"\nChecking: \"{user_statement}\"")
    # Step 1: Retrieve verses semantically similar to the user's statement.
    all_retrieved_verses = retrieve_top_verses(user_statement, k_retrieval=7)
    if not all_retrieved_verses:
        default_return["explanation"] = "No verses retrieved."; return default_return

    # Store the top retrieved verse as a potential reference.
    default_return["top_retrieved_verse_text"] = all_retrieved_verses[0]["text"]
    default_return["retrieved_reference"] = all_retrieved_verses[0]["location"]

    # Step 2: Prepare the context for the LLM using the top 5 retrieved verses.
    verses_for_llm = all_retrieved_verses[:5]
    verses_section = "Relevant Ramayana Verses:\n"
    print("\nVerses for LLM (top 5 of retrieved):")
    for i, v_info in enumerate(verses_for_llm):
        verses_section += f"{i+1}. ({v_info['location']}): \"{v_info['text']}\"\n"
        print(f"{i+1}. ({v_info['location']}): \"{v_info['text'][:150].replace(chr(10), ' ')}...\" (Score: {v_info['score']:.4f})")

    # Step 3: Construct the prompt for the LLM.
    prompt_intro = "You are an expert scholar of the Valmiki Ramayana."
    prompt_task = ("Analyze the 'User Statement' based *only* on the 'Relevant Ramayana Verses' provided. "
                   "Your answer MUST start with one of three words: TRUE, FALSE, or UNDETERMINED. "
                   "After the word, provide a brief explanation for your decision.")
    # Combine all parts into the final prompt using the instruction format for Mistral models.
    full_prompt = f"[INST] {prompt_intro}\n{prompt_task}\n\n{verses_section}\nUser Statement: \"{user_statement}\"\n\nAnswer: [/INST]"

    # Step 4: Query the LLM and get its response.
    try:
        output = llm_model(full_prompt, max_tokens=150, stop=["</s>", "[/INST]"], temperature=0.1, echo=False)
        llm_response = output['choices'][0]['text'].strip()
        default_return["llm_raw_response"] = llm_response
        print(f"\nLLM Raw Response: {llm_response}")
    except Exception as e:
        default_return["explanation"] = f"LLM generation error: {e}"; return default_return

    # Step 5: Parse the LLM's response.
    # Use regex to find if the response starts with TRUE, FALSE, or UNDETERMINED.
    match = re.match(r"^\s*(TRUE|FALSE|UNDETERMINED)\b", llm_response, re.IGNORECASE)
    if match:
        decision = match.group(1).upper()
        explanation = llm_response[match.end():].strip().lstrip(':').strip()
        # Convert the decision string to a boolean (or None for UNDETERMINED).
        parsed_result = {"TRUE": True, "FALSE": False}.get(decision)
        print(f"Parsed Decision: {decision}. Explanation: {explanation}")

        # Special logic for TRUE statements: Find the chronologically earliest verse
        # among the retrieved ones that supports the claim. This helps provide the
        # most fundamental reference.
        if parsed_result is True:
            best_ref_info_for_true = verses_for_llm[0]
            current_best_kanda_idx = float('inf')
            for v_info_llm in verses_for_llm:
                kanda_name = get_kanda_name_from_location_string(v_info_llm['location'])
                if kanda_name:
                    # Find the index of the kanda in our canonical order.
                    for k_idx, canonical_kanda in enumerate(KANDA_ORDER):
                        if canonical_kanda.lower().startswith(kanda_name.lower()):
                            if k_idx < current_best_kanda_idx:
                                # Ensure there is some word overlap to avoid spurious matches.
                                if len(set(user_statement.lower().split()).intersection(set(v_info_llm['text'].lower().split()))) > 1:
                                    current_best_kanda_idx = k_idx
                                    best_ref_info_for_true = v_info_llm
                            break
            # Update the reference to this chronologically earlier verse.
            default_return["retrieved_reference"] = best_ref_info_for_true["location"]
            default_return["top_retrieved_verse_text"] = best_ref_info_for_true["text"]
            print(f"Preferred Reference (due to TRUE and Kanda order): {best_ref_info_for_true['location']}")
        
        default_return["result_bool"] = parsed_result
        default_return["explanation"] = explanation
    else:
        # If the LLM response doesn't follow the format, mark it as an error.
        default_return["explanation"] = "LLM response format unexpected."
        print(f"LLM response format unexpected: {llm_response}")
        
    return default_return

def log_results_to_csv(log_entries_list: list):
    """
    Appends a list of fact-checking results to the output CSV file.
    Creates the file and header if it doesn't exist.
    """
    if not log_entries_list: return

    # Define the columns for the output CSV.
    output_cols = [
        "ID", "Statement", "Truth Value (actual)", 
        "Truth Value (predicted)", "Reasoning & Analysis (for incorrect prediction)"
    ]
    # Create a DataFrame from the list of result dictionaries.
    log_df = pd.DataFrame(log_entries_list)
    # Ensure the DataFrame has all the required columns in the correct order.
    log_df = log_df.reindex(columns=output_cols, fill_value="")

    # Check if the file already exists to decide whether to write the header.
    file_exists = os.path.isfile(LOG_CSV_PATH) and os.path.getsize(LOG_CSV_PATH) > 0
    # Append the data to the CSV file without writing the index.
    log_df.to_csv(LOG_CSV_PATH, mode='a', header=not file_exists, index=False, encoding='utf-8')

# --- Main execution block ---
if __name__ == "__main__":
    # Run the setup function to initialize everything.
    setup_models_and_data()

    print("\nRamayana Fact Checker (GGUF Version)")
    print(f"Logging results to: {LOG_CSV_PATH}")
    print("Type your statement, 'file: <path_to_your_csv.csv>', or 'q' to quit.\n")

    current_id_counter = 1

    # Start the main interactive loop.
    while True:
        user_input = input("\nEnter statement (or 'file: <path>' or 'q'): ").strip()
        if user_input.lower() == 'q': break
        if not user_input: continue

        stmts_to_process = []
        is_file_input = user_input.lower().startswith("file:")

        # Handle file input for batch processing.
        if is_file_input:
            fpath = user_input[len("file:"):].strip()
            try:
                input_df = pd.read_csv(fpath)
                if "Statement" not in input_df.columns:
                    print(f"ERROR: Input CSV must have a 'Statement' column."); continue
                
                # Prepare a list of statements to process from the file.
                for _, row in input_df.iterrows():
                    stmts_to_process.append({
                        "id": row.get("ID", current_id_counter),
                        "text": str(row["Statement"]),
                        "actual": str(row.get("Actual Truth", "N/A")) # Optional ground truth column.
                    })
                    if "ID" not in input_df.columns: current_id_counter += 1
                print(f"Processing {len(stmts_to_process)} statements from file: {fpath}...")
            except Exception as e:
                print(f"Error processing file {fpath}: {e}"); continue
        else:
            # Handle a single statement from the command line.
            stmts_to_process.append({"id": current_id_counter, "text": user_input, "actual": "N/A"})
            current_id_counter += 1
        
        batch_log_entries = []
        # Process each statement in the list (will be 1 for interactive mode).
        for stmt_info in stmts_to_process:
            s_id, s_text, s_actual = stmt_info["id"], stmt_info["text"], stmt_info["actual"]
            if not s_text or pd.isna(s_text): continue

            # Run the core fact-checking logic.
            output_dict = is_ramayana_statement_correct(s_text)

            # Convert the boolean prediction to a string for logging.
            predicted_truth_str = "TRUE" if output_dict["result_bool"] is True else \
                                  ("FALSE" if output_dict["result_bool"] is False else "UNDETERMINED")
            
            # Prepare the log entry for the CSV file.
            reasoning_analysis = ""
            # If an "Actual Truth" was provided and it doesn't match the prediction,
            # populate the analysis column with diagnostic information.
            if s_actual != "N/A" and predicted_truth_str.upper() != s_actual.upper():
                reasoning_analysis = (
                    f"PREDICTION MISMATCH. "
                    f"LLM Explanation: {output_dict.get('explanation', 'N/A')}. "
                    f"Top Retrieved Verse ({output_dict.get('retrieved_reference', 'N/A')}): \"{output_dict.get('top_retrieved_verse_text', 'N/A')}\""
                )
            
            log_entry = {
                "ID": s_id,
                "Statement": s_text,
                "Truth Value (actual)": s_actual,
                "Truth Value (predicted)": predicted_truth_str,
                "Reasoning & Analysis (for incorrect prediction)": reasoning_analysis
            }
            batch_log_entries.append(log_entry)

            # Print a summary to the console for interactive use.
            if not is_file_input or len(stmts_to_process) == 1:
                print(f"\nResult for ID {s_id}: {predicted_truth_str}")
                if output_dict["explanation"] != "N/A":
                    print(f"Explanation: {output_dict['explanation']}")
                if output_dict["top_retrieved_verse_text"] != "N/A":
                    print(f"Ref. Verse ({output_dict['retrieved_reference']}): \"{output_dict['top_retrieved_verse_text'][:200]}...\"")
                print("-" * 30)

        # After processing the batch, log all results to the CSV file.
        if batch_log_entries:
            log_results_to_csv(batch_log_entries)
            if is_file_input:
                print(f"\nFinished processing file. {len(batch_log_entries)} results appended to {LOG_CSV_PATH}.")

    # Explicitly free the model resources before the script exits to prevent errors.
    if llm_model is not None:
        print("\nFreeing LLM resources...")
        del llm_model

    print("Goodbye!")
