# IYD 2025 Hackathon - Ramayana Fact-Checking System

**Team Name:** FactSetu

**Bridging User Belief with Ramayana Wisdom**

[![Built with](https://img.shields.io/badge/Built%20with-Python-blueviolet)](https://www.python.org/)
[![Uses](https://img.shields.io/badge/Uses-SBERT%20%7C%20FAISS%20%7C%20Mistral%20LLM-ff69b4)](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
[![Model Format](https://img.shields.io/badge/Model%20Format-GGUF-orange)](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
[![Data Source](https://img.shields.io/badge/Data%20Source-Valmiki%20Ramayana-red)](https://www.sacred-texts.com/hin/rama/index.htm)

## Problem - Solution - Goal (Hackathon Theme)

### Problem
Difficulty in verifying statements about complex, large texts like the *Valmiki Ramayana* against the source text quickly and reliably. Traditional methods are manual and time-consuming.

### Solution
Develop an automated system that leverages **Semantic Search** (using Sentence Embeddings and FAISS) and **Large Language Models (LLMs)** to fact-check user-submitted statements against a digitized version of the *Valmiki Ramayana*.

### Goal
To create a system that can take user statements, efficiently search the *Valmiki Ramayana* text for relevant verses, and use an LLM to determine the truthfulness of the statement strictly based on the retrieved scriptural context. The output should be a clear decision (e.g., `TRUE`, `FALSE`, `NOT RELEVANT`) along with supporting textual evidence (verses).

## Project Overview

The Ramayana Fact-Checking System provides an automated way to verify claims related to the *Valmiki Ramayana*. It combines modern NLP techniques with a Large Language Model to compare user statements against the scripture.

The project primarily consists of two script variants:
*   `Fact_Checker.py`: A simplified version for batch processing from a CSV input, saving results to a CSV output.
*   `Fact_Checker_Detailed.py`: A more informative version, likely for interactive use or more verbose logging, providing verse text, location details, explanations, and raw LLM output.

## Detailed Functionality (`Fact_Checker_Detailed.py`)

This script variant is designed to provide comprehensive output for each fact-checking query. Its core algorithm follows these steps:

1.  **Setup & Data Loading:**
    *   Load the Ramayana dataset (presumably from a CSV containing verses and their metadata).
    *   Initialize a Sentence Transformer model (e.g., `all-mpnet-base-v2`) to generate vector embeddings for text.
    *   Load or compute embeddings for all verses in the Ramayana dataset. These embeddings are typically cached for faster access.
    *   Build a FAISS index on the verse embeddings to enable rapid approximate nearest neighbor search.
    *   Load a Large Language Model in GGUF format (e.g., a finetuned Mistral-7B-Instruct) which will perform the fact-checking logic.

2.  **Verse Retrieval (Semantic Search):**
    *   Receive a user statement.
    *   Generate a vector embedding for the user statement using the Sentence Transformer.
    *   Query the FAISS index with the statement embedding to find the top-K (e.g., 7) most semantically similar verses from the Ramayana text. Retrieve the verse text and their associated metadata (Kanda, Sarga, Shloka).

3.  **LLM-based Fact-Checking:**
    *   Select a subset of the retrieved verses (e.g., the top 5) and format them clearly, including their location details (Kanda, Sarga, Shloka).
    *   Construct a specific prompt for the loaded GGUF LLM. This prompt instructs the LLM to act as an expert on the *Valmiki Ramayana* and to determine if the user's statement is strictly supported (TRUE), contradicted (FALSE), or cannot be confirmed/denied (UNDETERMINED) *based solely on the provided subset of verses*. The prompt also asks for an explanation.
    *   Pass the constructed prompt and the retrieved verses to the LLM and obtain its response.

4.  **Response Parsing & Reference Selection:**
    *   Parse the LLM's raw text response to extract the explicit decision (TRUE/FALSE/UNDETERMINED) and the generated explanation.
    *   If the LLM's decision is TRUE, a heuristic is applied to select the "best" supporting verse to present as the primary reference. This heuristic might prioritize verses from earlier Kandas (based on a defined order like `KANDA_ORDER`) or verses that have keyword overlap with the original user statement.
    *   If the decision is FALSE or UNDETERMINED, the top verse retrieved by FAISS (or another relevant verse) is typically chosen as the reference.

5.  **Logging & Output:**
    *   Log the entire query process, including the original user statement, the LLM's final decision, the generated explanation, the text and location of the selected reference verse, and potentially the full raw LLM response. This logging can be saved to a file (like a CSV).
    *   Present the results to the user or output them to a console/file, depending on whether the script is run interactively or in batch mode.

## System Pipeline (`Fact_Checker.py`)

The `Fact_Checker.py` script implements the core fact-checking pipeline primarily for batch processing. The sequence of operations is as follows:

1.  **Input:** The script starts by taking the path to an input CSV file (e.g., `input.csv`) via command line argument. This file is expected to contain a column named "Statement" with the user claims to be fact-checked.
2.  **Input Validation:** It validates the input file, checking for its existence and the presence of the required "Statement" column.
3.  **Resource Setup:** The `setup_models_and_data` function is called once. This function handles:
    *   Downloading necessary assets (Ramayana data, embeddings, GGUF LLM model) from a specified Hugging Face repository (`dishasahu/ramayana-fact-checker-assets`).
    *   Loading the Ramayana data and setting up the Sentence Transformer and FAISS index for efficient retrieval.
    *   Loading the GGUF LLM model into memory.
4.  **Iterate Statements:** The script then iterates through each statement found in the input CSV file.
5.  **Fact-Checking Process (`get_llm_decision`):** For each statement, the `get_llm_decision` function is invoked. This function encapsulates the steps described in the "Detailed Functionality" section:
    *   It performs the semantic search (`retrieve_top_verses`) using the FAISS index to find relevant verses.
    *   It constructs the LLM prompt using the statement and the retrieved verses.
    *   It sends the prompt to the loaded GGUF LLM.
    *   It parses the LLM's response to determine the decision (TRUE/FALSE/UNDETERMINED) and extracts explanations and references.
6.  **Map Decision:** The parsed decision from the LLM is mapped to a standard output string like "TRUE", "FALSE", or "NOT RELEVANT" (corresponding to UNDETERMINED).
7.  **Collect Results:** The result for each statement (including the statement itself, the decision, explanation, and reference) is stored.
8.  **Save Output:** After processing all statements in the input CSV, all collected results are saved to a new CSV file named `prediction_output.csv`.
9.  **Cleanup:** The script performs necessary cleanup, typically involving releasing the resources used by the loaded LLM.

## How to Run (`Fact_Checker.py`)

1.  **Prerequisites:** Ensure you have Python and the necessary libraries installed. You will also need the GGUF model and other assets, which the script is designed to download automatically from the specified Hugging Face repository.
2.  **Input File:** Create an input CSV file (e.g., `my_statements.csv`) in the same directory as the script. This file must contain a header row and a column specifically named `"Statement"`.
    Example `my_statements.csv`:
    ```csv
    ID,Statement
    1,"Ram is the eldest son of King Dasharatha."
    2,"Hanuman is born to Anjana and Kesari with blessings from the wind god, Vayu."
    3,"Demoness Shurpanakha played a pivotal role in inciting the conflict between Rama and Ravana."
    4,"Rama and Sita had children named Luv and Kush."
    ```
3.  **Run from Terminal:** Open your terminal or command prompt, navigate to the directory where you saved `Fact_Checker.py` and your input CSV, and run the script using the following command format:

    ```bash
    python Fact_Checker.py my_statements.csv
    ```
4.  **Output:** The script will process the statements and save the results to a file named `prediction_output.csv` in the same directory.
## This diagram provides a high-level view:
```text
+---------------------+
|    Input: CSV       |
|    (input.csv)      |
+----------+----------+
           |
           v
+---------------------+
|       main()        |
+----------+----------+
           |
           v
+---------------------+
|  Validate Input:    |
|  - File Exists      |
|  - 'Statement' Col  |
+----------+----------+
           |
           v
+---------------------+
| setup_models_and_data|
|    (One-Time Setup) |
+----------+----------+
           |
           v
+---------------------+
|  Loop: For Each     |
|    Statement        |
|  in Input CSV       |
+----------+----------+
           |
           v
+---------------------+
|  get_llm_decision() |
|  (for current stmt) |
+----------+----------+
           |
           v
+---------------------+
|  retrieve_top_verses|
|  (Semantic Search)  |
+----------+----------+
           |
           v
+---------------------+
| Construct LLM Prompt|
| (Statement + Verses)|
+----------+----------+
           |
           v
+---------------------+
|  Feed Prompt to LLM |
|     (Mistral GGUF)  |
+----------+----------+
           |
           v
+---------------------+
|  Parse LLM Response |
|  (TRUE/FALSE/UNDET.)|
+----------+----------+
           |
+----------+----------+ <----+
|Map Decision to String|     |
|("TRUE"/"FALSE"/    |     |
|"NOT RELEVANT")     |     |
+----------+----------+     |
           |                |
           +----------------+ (Loop continues for next statement)
           |
(After all statements processed)
           v
+---------------------+
|  Save all results to|
| prediction_output.csv|
+----------+----------+
           |
           v
+---------------------+
|   Clean Up LLM      |
|  (Release Memory)   |
+---------------------+
           |
           v
+---------------------+
|     Process         |
|     Complete        |
+---------------------+

FactSetu is still under active development. While the core pipeline is functional and produces reliable verdicts, we're working on improvements to enhance accuracy, scalability, and user accessibility.
