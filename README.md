# Belief and LLMs

This repository is for my postdoc project, *Belief and LLMs*, and this is the first version of the whole project.

## Research Goals

The ultimate goal of this research is to completely understand Large Language Models (LLMs)' internal structure and investigate how much the encoded beliefs and stereotypes in LLMs reflect real-world human behavior. Specifically, we:

- Extract latent vector representations of demographic personas (race, political affiliation, gender, education, religion, generation)
- Compare LLM-encoded beliefs against real-world data from the General Social Survey (GSS)
- Study the additivity of demographic traits in model representations
- Develop steering vectors to controllably modify model outputs

## Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 0: DATA CURATION (Local)                                         │
│  0_curate_GSS.ipynb, 0_1_curate_GSS_exp2.ipynb                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
      ┌─────────────────────────────┼─────────────────────────────┐
      ▼                             ▼                             ▼
┌───────────────┐      ┌────────────────────────┐      ┌──────────────────┐
│gss_cleaned.csv│      │gss_correlation_pairs.csv│     │gss_extraction_   │
│               │      │(722 demographic pairs) │      │datasets.json     │
└───────────────┘      └────────────────────────┘      └────────┬─────────┘
                                    │                           │
┌───────────────────────────────────┼───────────────────────────┼─────────┐
│  PHASE 1: VECTOR EXTRACTION (Google Colab + GPU)              │         │
│  1_extract_persona_vectors.ipynb                              │         │
└───────────────────────────────────┼───────────────────────────┼─────────┘
                                    │                           │
                                    ▼                           ▼
                       ┌────────────────────────────────────────────────┐
                       │  Meta-Llama-3-8B-Instruct (4-bit quantized)    │
                       │  Forward Hook @ Layers 5-20 → 4096-dim vectors │
                       │  Steering Vector = mean(X+) - mean(X-)         │
                       └────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
┌──────────────────────────┐            ┌─────────────────────────────────┐
│ demographic_vectors_     │            │ demo_vectors_similarity_        │
│ layer{N}.pt (per layer)  │            │ results.csv                     │
└────────────┬─────────────┘            └────────────────┬────────────────┘
             │                                           │
┌────────────┼───────────────────────────────────────────┼────────────────┐
│  PHASE 2: STEERING & VALIDATION (Google Colab + GPU)   │                │
│  2_simulate_steering_vectors.ipynb                     │                │
└────────────┼───────────────────────────────────────────┼────────────────┘
             │                                           │
             ▼                                           ▼
┌──────────────────────────┐            ┌─────────────────────────────────┐
│ Steering Experiments     │            │ Statistical Validation          │
│ • Inject @ Layers 5-20   │            │ • LLM vs GSS correlation: r=0.74│
│ • Strength: ±2 to ±3     │            │ • Amplification factor: 1.38    │
│ • Policy questions       │            │ • Additivity test: cos_sim >0.89│
└────────────┬─────────────┘            └─────────────────────────────────┘
             │
┌────────────┼───────────────────────────────────────────────────────────┐
│  PHASE 3: CORRELATION INVESTIGATION (Google Colab + GPU)               │
│  3_investigate_correlations.ipynb                                      │
└────────────┼───────────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Delta Magnitude Analysis                                               │
│ • Δ = magnitude(layer N) - magnitude(layer N-1)                        │
│ • Identifies layer-specific contribution (vs accumulated signal)       │
│ • Correlates Δ with steering effectiveness to find optimal layer       │
└────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
belief_and_llms_v0/
├── data/
│   ├── GSS.xlsx                        # General Social Survey source data
│   ├── gss_cleaned.csv                 # Cleaned demographics subset
│   ├── gss_correlation_pairs.csv       # Phi-coefficient correlations
│   ├── gss_extraction_datasets.json    # Contrastive prompt pairs
│   └── demo_vectors_similarity_results.csv
├── 0_curate_GSS.ipynb                  # Phase 0: Data curation
├── 0_1_curate_GSS_exp2.ipynb           # Phase 0: Additivity experiment setup
├── 1_extract_persona_vectors.ipynb     # Phase 1: Vector extraction (Colab)
├── 2_find_corr_GSS.ipynb               # Phase 2: Correlation analysis
├── 2_simulate_steering_vectors.ipynb   # Phase 2: Steering experiments (Colab)
└── 3_investigate_correlations.ipynb    # Phase 3: Correlation investigation (Colab)
```

## Technical Details

| Component | Specification |
|-----------|---------------|
| Model | Meta-Llama-3-8B-Instruct |
| Quantization | 4-bit (bitsandbytes) |
| Vector Extraction Layer | Layers 5-20 |
| Steering Injection Layer | Layers 5-20 (same as extraction) |
| Hidden Dimension | 4096 |

## Key Findings

- **Alignment:** Strong correlation (r = 0.74) between LLM-encoded beliefs and real-world GSS data
- **Amplification:** LLM exaggerates demographic stereotypes by ~38% (slope = 1.38)
- **Additivity:** Composite personas (e.g., "Black Democrat") align well with summed component vectors (cosine similarity > 0.89)
- **Controllability:** Steering vectors effectively shift model outputs on policy questions

## Delta Magnitude Analysis

Due to residual connections in transformers, absolute magnitude accumulates across layers. To find the optimal steering layer, we analyze **delta magnitude**:

```
Δ(layer N) = magnitude(N) - magnitude(N-1)
```

| Metric | What it measures |
|--------|------------------|
| Absolute magnitude | Total accumulated signal up to layer N |
| Delta magnitude (Δ) | How much layer N specifically contributes |

The layer with highest Δ is where the model adds the most demographic-specific information—potentially the best target for steering injection.

## Requirements

- PyTorch
- HuggingFace Transformers
- BitsAndBytes
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn

GPU-intensive notebooks (Phase 1 & 2) are designed to run on Google Colab.
