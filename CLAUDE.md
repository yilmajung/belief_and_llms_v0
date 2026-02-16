# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The ultimate goal of this research is to completely understand how Large Language Models (LLMs) internally embed demographic characteristics and investigate their correlations. By steering the demographic vectors extracted from Meta-Llama-3-8B-Instruct hidden states, we experiment with precisely and minutely controlling LLM responses based on demographic characteristics.

Inspired by "The Assistant Axis" (Lu et al., 2026), we are extending the project to map out a low-dimensional "Demographic Space" via PCA on extracted demographic vectors, identify interpretable axes of demographic variation (analogous to the Assistant Axis), and study how models drift along demographic axes during multi-turn conversations on politically charged topics.

### Key References

- **The Assistant Axis** (Lu et al., 2026): Extracts activation directions for 275 character archetypes, discovers a dominant "Assistant Axis" via PCA, and uses activation capping to stabilize model persona. Key methodological inspirations:
  - PCA on extracted vectors to discover interpretable low-dimensional structure
  - LLM judge filtering to ensure response quality before computing mean vectors
  - Activation capping (`h ← h − v · min(⟨h, v⟩ − τ, 0)`) for bounded steering
  - Multi-turn persona drift tracking via activation projections

## Architecture

### Processing Pipeline

The project follows a numbered notebook sequence where each phase builds on previous outputs:

1. **Phase 0 - Data Curation** (`0_curate_GSS.ipynb`, `0_1_curate_GSS_exp2.ipynb`)
   - Loads General Social Survey (GSS) data from `data/GSS.xlsx`
   - Creates demographic mappings for 25 variables: race, party, sex, degree, religion, political views, generation, marital status, age group, children, immigration generation, region (grew up & current), family income, urbanity, occupation (SOC), industry (NAICS), happiness, health, life excitement, job satisfaction, social class, financial satisfaction, gun ownership, belief in God
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

5. **Phase 3.1 - Contrastive Steering** (`3_1_contrastive_steering.ipynb`)
   - Tests contrastive steering vectors: `v_contrastive = v_Republican - v_Democrat`
   - Original vectors encode "X vs generic" → all deltas positive
   - Contrastive vectors encode "X vs Y" → oppositional effects (some positive, some negative)
   - Compares original vs contrastive steering effectiveness
   - Analyzes which layer produces clearest oppositional patterns

6. **Phase 4 - Multi-Dimensional Steering** (`4_multi_dimensional_steering.ipynb`)
   - Applies multiple demographic steering vectors simultaneously
   - Tests composite persona construction (e.g., "Black Democrat")

#### Planned Phases (inspired by "The Assistant Axis")

7. **Phase 5 - Demographic Space Mapping** (planned)
   - Stack all ~119 demographic vectors → PCA to discover "Demographic Space"
   - Identify PC1 as the dominant axis of demographic variation (hypothesized: liberal-conservative)
   - Analyze interpretable structure of higher PCs (education/generation? urban/rural?)
   - Compare angular relationships in PCA space with phi-coefficient correlations from GSS

8. **Phase 6 - LLM Judge Filtering** (planned)
   - Use an LLM judge to score whether responses genuinely express the target demographic
   - Filter out weak activations before computing `mean(X+) - mean(X-)` for cleaner vectors
   - Re-extract demographic vectors with quality-filtered responses

9. **Phase 7 - Demographic Drift in Multi-Turn Conversations** (planned)
   - Track model's projection along demographic axes turn-by-turn in politically charged conversations
   - Study whether the model drifts toward particular demographic profiles on charged topics
   - Use activation capping to constrain demographic drift within a bounded range

10. **Phase 8 - Activation Capping for Demographic Control** (planned)
    - Implement activation capping: `h ← h − v · min(⟨h, v⟩ − τ, 0)`
    - Cap activations along demographic axes to prevent the model from drifting too far
    - Compare with additive steering (±2-3 strength) for stability and controllability

### Key Technical Details

- **Model:** Meta-Llama-3-8B-Instruct with 4-bit quantization (bitsandbytes)
- **Demographic Groups:** ~119 total across 25 variables:
  - Original (7): race, party, sex, degree, religion (expanded to 7 categories), political views, generation
  - New demographics (11): marital status, age group, children, immigration generation, region grew up, family income, current region, urbanity, occupation (SOC major groups), industry (NAICS sectors), military
  - Attitudes/values (7): happiness, health, life excitement, job satisfaction, social class, financial satisfaction, gun ownership, belief in God
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
