# Supported Strategies in Speech-to-Text Benchmark

This document identifies which strategies from the **Unified Evaluation Workflow** are natively supported by the Speech-to-Text Benchmark harness. A strategy is considered "supported" only if the harness provides it out-of-the-box in its full installation—meaning that once the harness is fully installed via `pip install -r requirements.txt`, the strategy can be executed directly without implementing custom modules or integrating external libraries.

---

## Phase 0: Provisioning (The Runtime)

### Step A: Harness Installation

**Supported Strategies:**

- ✅ **Strategy 1: Git Clone** - The harness can be cloned from its GitHub repository and installed manually from source.
- ✅ **Strategy 2: PyPI Packages** - The harness is installed via pip with `pip install -r requirements.txt`, which includes all necessary Python packages from PyPI.

**Unsupported Strategies:**

- ❌ **Strategy 3: Node Package** - Not applicable; this is a Python-based harness.
- ❌ **Strategy 4: Binary Packages** - The harness does not provide standalone executable binaries.
- ❌ **Strategy 5: Container Images** - The harness does not provide prebuilt Docker or OCI container images.

### Step B: Credential Configuration

**Supported Strategies:**

- ✅ **Strategy 1: Model API Authentication** - The harness supports configuring API keys via command-line arguments and environment variables for remote inference with:
  - Amazon Transcribe (AWS credentials via `--aws-profile` and AWS environment)
  - Azure Speech-to-Text (`--azure-speech-key` and `--azure-speech-location`)
  - Google Speech-to-Text (`--google-application-credentials` environment variable)
  - IBM Watson Speech-to-Text (`--watson-speech-to-text-api-key` and `--watson-speech-to-text-url`)
  - Picovoice Cheetah/Leopard (`--picovoice-access-key`)

- ✅ **Strategy 2: Artifact Repository Authentication** - The harness supports downloading models from:
  - Hugging Face Hub (via `huggingface-hub` dependency for Whisper models)
  - Model repositories via OpenAI Whisper library (automatically downloads model weights when needed)

**Unsupported Strategies:**

- ❌ **Strategy 3: Evaluation Platform Authentication** - The harness does not support submitting results to evaluation platforms or leaderboard APIs.

---

## Phase I: Specification (The Contract)

### Step A: SUT Preparation

**Supported Strategies:**

- ✅ **Strategy 1: Model-as-a-Service (Remote Inference)** - The harness natively supports evaluating remote API-based speech-to-text models:
  - Amazon Transcribe (batch and streaming)
  - Azure Speech-to-Text (batch and real-time streaming)
  - Google Speech-to-Text (batch and streaming, with enhanced variants)
  - IBM Watson Speech-to-Text

- ✅ **Strategy 2: Model-in-Process (Local Inference)** - The harness natively supports evaluating locally-run models with weights loaded into memory:
  - OpenAI Whisper (all variants: tiny, base, small, medium, large, large-v2, large-v3)
  - Picovoice Cheetah (streaming)
  - Picovoice Leopard (batch)

**Unsupported Strategies:**

- ❌ **Strategy 3: Non-Parametric Algorithms (Deterministic Computation)** - The harness does not evaluate non-parametric algorithms like ANN indexes, BM25, or signal processing pipelines.
- ❌ **Strategy 4: Interactive Agents (Sequential Decision-Making)** - The harness does not support evaluating interactive agents or sequential decision-making systems.

### Step B: Benchmark Preparation (Inputs)

**Supported Strategies:**

- ✅ **Strategy 1: Benchmark Data Preparation (Offline)** - The harness natively supports loading predefined benchmark datasets:
  - LibriSpeech (test-clean and test-other)
  - TED-LIUM
  - Common Voice
  - Multilingual LibriSpeech (MLS)
  - VoxPopuli
  - Fleurs (with download script provided)
  
  The harness loads audio files and reference transcripts from local filesystem paths specified via `--dataset-folder`.

**Unsupported Strategies:**

- ❌ **Strategy 2: Synthetic Data Generation (Generative)** - The harness does not support generating synthetic test data, input perturbation, or test augmentation.
- ❌ **Strategy 3: Simulation Environment Setup (Simulated)** - Not applicable to speech-to-text evaluation; the harness does not use simulated environments.
- ❌ **Strategy 4: Production Traffic Sampling (Online)** - The harness does not support sampling real-world production traffic for evaluation.

### Step C: Benchmark Preparation (References)

**Supported Strategies:**

- ✅ **Strategy 1: Ground Truth Preparation** - The harness pre-loads ground truth reference transcripts from dataset files. For latency benchmarking, it also supports loading word-level timing alignments. The repository includes a utility script (`script/generate_alignments.py`) to generate these alignments using the Montreal Forced Aligner (MFA), though MFA itself must be installed separately via conda as documented in `script/README.md`.

**Unsupported Strategies:**

- ❌ **Strategy 2: Judge Preparation** - The harness does not support setting up or using LLM-based judges, trained reward models, or other model-based evaluators for subjective assessment.

---

## Phase II: Execution (The Run)

### Step A: SUT Invocation

**Supported Strategies:**

- ✅ **Strategy 1: Batch Inference** - The harness executes batch inference by processing multiple audio files through a single SUT instance sequentially. It supports parallel processing using `ProcessPoolExecutor` with configurable `--num-workers` to process datasets efficiently.

**Unsupported Strategies:**

- ❌ **Strategy 2: Arena Battle** - The harness does not support concurrent execution of the same input across multiple SUTs for paired comparison.
- ❌ **Strategy 3: Interactive Loop** - The harness does not support stateful step-by-step execution through interactive environments.
- ❌ **Strategy 4: Production Streaming** - The harness does not support continuous processing of live production traffic or real-time drift monitoring.

---

## Phase III: Assessment (The Score)

### Step A: Individual Scoring

**Supported Strategies:**

- ✅ **Strategy 1: Deterministic Measurement** - The harness natively computes deterministic text-based metrics:
  - **Word Error Rate (WER)** - Edit distance between predicted and reference word sequences
  - **Punctuation Error Rate (PER)** - Punctuation-specific error calculation following [Meister et al.](https://arxiv.org/abs/2310.02943)
  
  These are computed using exact algorithmic calculations (edit distance via the `editdistance` library).

- ✅ **Strategy 4: Performance Measurement** - The harness measures computational efficiency metrics:
  - **Real-Time Factor (RTF)** / Core-Hour - CPU time required to process audio (via `process_sec()` and `audio_sec()` tracking)
  - **Word Emission Latency** - For streaming engines, measures the delay from word completion to transcription emission (via `benchmark_latency.py`)
  - **Model Size** - Reports aggregate model size in MB (documented in README for local engines)

**Unsupported Strategies:**

- ❌ **Strategy 2: Embedding Measurement** - The harness does not compute semantic similarity metrics using embeddings (e.g., BERTScore, sentence embeddings).
- ❌ **Strategy 3: Subjective Measurement** - The harness does not use LLMs or other models as judges to assess subjective quality attributes or perform pairwise comparisons.

### Step B: Aggregate Scoring

**Supported Strategies:**

- ✅ **Strategy 1: Distributional Statistics** - The harness aggregates per-instance scores into dataset-level metrics:
  - Computes average error rates across all test instances
  - Calculates overall WER/PER by summing errors and tokens across workers
  - Averages performance metrics (RTF, latency) across samples
  - The `plot_results.py` script computes averages across multiple datasets

**Unsupported Strategies:**

- ❌ **Strategy 2: Uncertainty Quantification** - The harness does not compute confidence intervals, bootstrap resampling, or other uncertainty estimates around aggregate metrics.

---

## Phase IV: Reporting (The Output)

### Step A: Insight Presentation

**Supported Strategies:**

- ✅ **Strategy 4: Chart Generation** - The harness provides visualization capabilities via `plot_results.py`:
  - Bar charts comparing error rates across engines and datasets
  - Performance comparison plots (WER, PER, latency, Core-Hour)
  - Multi-dataset comparison visualizations
  - Results are saved as PNG files in the `results/plots/` directory

**Unsupported Strategies:**

- ❌ **Strategy 1: Execution Tracing** - The harness does not capture or display detailed step-by-step execution logs, intermediate states, or execution flow visualizations.
- ❌ **Strategy 2: Subgroup Analysis** - The harness does not support breaking down performance by demographic groups, data domains, or other stratification dimensions beyond the predefined dataset splits.
- ❌ **Strategy 3: Regression Alerting** - The harness does not automatically compare results against historical baselines or trigger performance degradation alerts.
- ❌ **Strategy 5: Dashboard Creation** - The harness does not provide interactive web interfaces or dashboards for browsing results.
- ❌ **Strategy 6: Leaderboard Publication** - The harness does not support submitting results to public or private leaderboards.

---

## Summary

The Speech-to-Text Benchmark harness is a **focused, minimalist evaluation framework** for speech-to-text engines. It provides native support for:

**Strengths:**
- Multiple installation methods (Git clone, PyPI)
- Comprehensive credential configuration for major cloud STT providers
- Both remote API and local model evaluation
- Standard speech benchmark datasets with ground truth transcripts
- Deterministic accuracy metrics (WER, PER) and performance metrics (latency, Core-Hour)
- Basic statistical aggregation and visualization via plotting

**Limitations:**
- No support for synthetic data generation, simulated environments, or production traffic
- No embedding-based or LLM-based subjective evaluation
- No uncertainty quantification (confidence intervals, bootstrap)
- No interactive dashboards, execution tracing, or leaderboard integration
- No regression detection or alerting capabilities

The harness excels at **traditional batch evaluation of speech-to-text models** with **deterministic metrics**, making it well-suited for standardized benchmarking but not for advanced evaluation scenarios requiring model-as-judge, uncertainty quantification, or production monitoring.
