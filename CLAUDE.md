# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The ultimate goal of this research is to completely understand how Large Language Models (LLMs)' internally embed demographic characteristics and investigate their correlations. Plus, by steering the demographic vectors extracted from Meta-Llama-3-8B-Instruct hidden states, I want to experiment if I can precisely and minutely control LLM reponses based on demographic characteristics.

## Architecture

### Processing Pipeline

The project follows a numbered notebook sequence where each phase builds on previous outputs:

1. **Phase 0 - Data Curation** (`0_curate_GSS.ipynb`, `0_1_curate_GSS_exp2.ipynb`)
   - Loads General Social Survey (GSS) data from `data/GSS.xlsx`
   - Creates demographic mappings for race, party, gender, degree, religion, political views, generation
   - Generates contrastive prompt pairs (X+ persona vs X- generic baseline) saved to `data/gss_extraction_datasets.json`
   - Computes phi-coefficient correlations saved to `data/gss_correlation_pairs.csv`

2. **Phase 1 - Vector Extraction** (`1_extract_persona_vectors.ipynb`)
   - Runs on Google Colab with GPU
   - Extracts 4096-dim latent vectors from Llama-3 Layers 5-20 using PyTorch forward hooks
   - Computes steering vectors as: `mean(X+) - mean(X-)` for each demographic
   - Outputs one vector file per layer: `gss_demographic_vectors_layer{N}.pt`

3. **Phase 2 - Steering Experiments** (`2_simulate_steering_vectors.ipynb`)
   - Runs on Google Colab with GPU
   - Tests steering vectors with injection strengths ±2-3
   - Validates on policy questions (abortion, gun control, climate, healthcare, taxes)
   - Tests layers 5-20 to find the best one

4. **Phase 3 - Correlation Investigation** (`3_investigate_correlations.ipynb`)
   - Investigates that when one demographic vector changes by a steering vector, how and in which direction other demographic vectors change

### Key Technical Details

- **Model:** Meta-Llama-3-8B-Instruct with 4-bit quantization (bitsandbytes)
- **Demographic Groups:** 34 total (race, party, sex, degree, religion, political views, generation)
- **Vector Extraction Layer:** Layer 5 to 20
- **Steering Injection Layer:** Layer 5 to 20 (same layer as extraction)
- **Hidden Dimension:** 4096

### Data Flow

```
GSS.xlsx → gss_cleaned.csv → gss_correlation_pairs.csv
                           → gss_extraction_datasets.json → demographic_vectors.pt → similarity_results.csv
```

## Development Environment

- GPU-intensive notebooks (1_*, 2_*) are designed to run on Google Colab with Google Drive mounting
- Local notebooks (0_*) run with standard Python environment
- Conda environment located in `.conda/`

## Dependencies

Core: PyTorch, HuggingFace Transformers, BitsAndBytes, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
