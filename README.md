# PEFT-Optimization Notebook

## Links
- 🤗 **[Live Demo (HF Space)](https://huggingface.co/spaces/matcov/study-card-generator)**
- 🤗 **[Model Weights](https://huggingface.co/dnagard/PEFT-optimization)**

## Project Overview
Fine-tuned Llama 3.2 1B on FineTome-100k using LoRA for KTH ID2223 Lab 2.
The model powers a Study Card Generator that creates flashcards from any topic.

## How to Run the UI Locally
```bash
pip install -r requirements.txt
python app.py
```

## Experiments

This section documents the fine-tuning experiments conducted so far. The goal of these early runs was to validate the full workflow—from data processing and LoRA training to GGUF export—while measuring GPU usage and training behavior under different configurations. All experiments were performed in Google Colab on a T4 GPU using 4-bit quantized base models and LoRA adapters.

---

### Experiment 1 — Minimal pipeline test (64 samples, 50 steps)

**Purpose:**
Confirm that the full training loop works on a very small dataset and completes quickly without running into formatting or tokenization issues.

**Setup:**

- Dataset: 64 samples (subset of FineTome-100k)
- Training steps: 1
- Batch size: 2, gradient accumulation 4
- Packing enabled
- No checkpointing needed at this scale

**Results:**

- Verified that data loading, formatting, LoRA application, and saving all behave as expected on very small data.  
- Useful for debugging and ensuring that the notebook was configured correctly before scaling up.
- No GGUF export for this run—used strictly for debugging.

---

### Experiment 2 — Small-scale smoke test (10k samples, 50 steps)

**Purpose:**
Run the full pipeline on a realistic dataset slice and confirm that everything works end-to-end with enough data to meaningfully stress the training loop.

**Setup:**

- Dataset: 10,000 samples
- Training steps: 50
- Batch size: 2, gradient accumulation 4
- Checkpoints every 10 steps
- Packing enabled

**Results:**

- Completed successfully.
- Produced a functional LoRA adapter and tokenizer outputs.
- Merged the LoRA adapter into a 16-bit base model.
- Converted the merged model to GGUF using llama.cpp (`q8_0`).
- Successfully pushed both the LoRA adapter and the GGUF model to Hugging Face Hub.
- Served as the first “real” run, however I forgot to note down timing and GPU usage.

---

### Experiment 3 — Extended training run (10k samples, 600 steps)

**Purpose:**
Evaluate longer training duration, measure GPU usage over time, and confirm training stability for deeper runs.

**Setup:**

- Dataset: 10,000 samples
- Training steps: 600
- Same training parameters as Experiment 2

**Performance Observations:**

- Training time: **1533.65 seconds (~25.56 minutes)**
- Peak reserved GPU memory: **2.275 GB**
- Additional LoRA-specific memory: **1.072 GB** (~7.27% of T4 capacity)
- No OOM issues or interruptions

**Outcomes:**

- Produced a higher-quality LoRA adapter.
- Confirmed that checkpoint resume logic functions correctly.
- Successfully merged the LoRA with a 16-bit base model.
- Converted to GGUF via llama.cpp using `q8_0`.
- Uploaded both LoRA weights and the GGUF file to Hugging Face Hub.
- Confirmed that longer fine-tuning runs fit comfortably within Colab limits.

---

## Summary

Across the first three experiments, the following components were successfully validated:

- End-to-end LoRA fine-tuning workflow
- Dataset formatting and ShareGPT standardization
- Stable training across short and extended runs
- Checkpointing and resume functionality
- Full export pipeline:
  - Merging LoRA → FP16
  - Converting FP16 → GGUF with llama.cpp
  - Uploading both LoRA and GGUF artifacts to Hugging Face

These runs establish a fully reproducible workflow that we can now expand with additional models, dataset sizes, and LoRA configurations.

## Task 2

Task 2 focuses on systematically improving our model along two primary axes:

1. Model-centric optimization (architecture & hyperparameters)
2. Data-centric scaling (more/better training data)

Additionally, we fine-tune a **second foundation model** (3B parameters) to compare quality and resource usage with a larger model.

To evaluate the models, we collected training metrics (loss, runtime, GPU memory) and manually evaluated all models on the same fixed set of 10 prompts:

- Explain how a VPN works in one short paragraph.
- List three differences between supervised and unsupervised learning.
- Write a polite professional email requesting feedback on a job interview.
- Give a recipe for a simple pasta dish using 5 ingredients or less.
- Translate this sentence to Swedish: The coffee is too hot to drink.
- What are two advantages and two disadvantages of renewable energy?
- Write a short Python function that reverses a string.
- Provide study tips for a student preparing for final exams.
- Explain what overfitting is using a real-world analogy.
- Who is Ada Lovelace? Provide 3 key points

### Zero-Shot Pre-Training Baseline (Reference Only)

**Purpose:**
Establish how well the pre-trained 1B Instruct model performs out-of-the-box before any fine-tuning.

**Setup:**

- Model: `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`
- No LoRA adapter / no additional training
- Evaluated only on the 10-prompt manual test set (`eval_No_LoRa`)

**Qualitative observations:**

- **Answers are generally coherent** and mostly on-topic.
- **System / meta text leakage**: every response begins with a ChatML-style header(`system\n\nCutting Knowledge Date...`) followed by the user prompt and assistant tag.
- **Swedish translation is incorrect**:

  > `Kaffe är för varm för att drinka.`
  
  which is grammatically wrong (should be *“Kaffet är för varmt för att dricka.”*).

- Other tasks (email, study tips, overfitting analogy, etc.) are acceptable but not especially polished or concise.

This run is treated as **T0**, the “no-training” baseline for all subsequent experiments.

---

### Experiment T1 — Fine-tuned baseline (1B, LoRA r=16, 10k, 600 steps)

**Purpose:**
Establish a baseline fine-tuned model to compare model-centric and data-centric improvements.

**Setup:**

- Run name: `baseline_1B_r16_10k_600steps`
- Model: `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`
- Dataset: `train[:10000]` from FineTome-100k
- LoRA: `r = 16`, `lora_alpha = 16`
- `max_steps = 600`

**Metrics (from `experiments_log.csv`):**

- Final training loss: **0.8918**
- Runtime: **2594.04 s** (~43.2 min)
- Peak GPU memory: **2.564 GB**
- LoRA-specific memory: **1.361 GB**

**Qualitative evaluation (`eval_baseline_1B_r16_10k_600steps`):**

- Outputs are more structured and somewhat more helpful than T0.
- System / meta-prompt leakage is **still present** in all answers.
- Swedish translation still fails (English sentence is repeated instead of translated), thus follows directions slightly worse than the base model.

T1 is the **fine-tuned reference point** for experiments T2 and T3.

---

### Experiment T2 — Model-centric scaling (1B, LoRA r=64, 10k, 600 steps)

**Purpose:**
Increase LoRA capacity to see whether a higher-rank adapter improves convergence and response quality.

**Setup:**

- Run name: `model_centric_1B_r64_10k_600steps`
- Same base model: `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`
- Dataset: `train[:10000]` (same as T1)
- LoRA: `r = 64`, `lora_alpha = 64`
- `max_steps = 600`

**Metrics:**

| Metric        | T1 (r=16) | T2 (r=64)    |
| ------------- | --------- | ------------ |
| Final loss    | 0.8918    | **0.8806** ⬇ |
| Runtime (s)   | 2594.04   | **2649.69**  |
| Peak GPU (GB) | 2.564     | **2.955**    |
| LoRA GPU (GB) | 1.361     | **1.752**    |

**Qualitative evaluation (`eval_model_centric_1B_r64_10k_600steps`):**

- Answers are **slightly more detailed and polished** than T1.
- Instruction following improves (clearer lists, better email, slightly richer explanations).
- System / meta-prompt leakage persists.
- Swedish translation still fails (English text repeated).

**Conclusion (model-centric):**

Increasing LoRA rank from 16 → 64 **improves convergence and perceived quality** for items within the training dataset, with only a modest increase in runtime and VRAM usage on a T4. However, the model also loses quality in certain areas not covered in the training dataset, like translations.

---

### Experiment T3 — Data-centric scaling (1B, LoRA r=64, 30k, 600 steps)

**Purpose:**
Test whether simply increasing the number of training examples improves downstream behavior.

**Setup:**

- Run name: `data_centric_1B_r64_30k_600steps`
- Model: `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`
- Dataset: `train[:30000]` (3× more data)
- LoRA: `r = 64`, `lora_alpha = 64` (same as T2)
- `max_steps = 600`

**Metrics:**

- Final training loss: **0.8835**
- Runtime: **2647.19 s** (~44.1 min)
- Peak GPU memory: **3.367 GB**
- LoRA-specific memory: **0.385 GB**


For comparison with T2 (same LoRA, different data size):

| Metric        | T2 (10k) | T3 (30k)                                              |
| ------------- | -------- | ----------------------------------------------------- |
| Final loss    | 0.8806   | **0.8835** (slightly worse, but different data slice) |
| Runtime (s)   | 2649.69  | **2647.19**                                           |
| Peak GPU (GB) | 2.955    | **3.367**                                             |


**Qualitative evaluation (`eval_data_centric_1B_r64_30k_600steps`):**

- Output style and quality are **very similar** to T2.
- System / meta-prompt leakage remains.
- Swedish translation still incorrect.
- No obvious qualitative gain over T2 despite 3× more data.

**Conclusion (data-centric):**

In this setup, **more data alone** did **not** fix the key failure modes or significantly improve behavior. Dataset composition / alignment likely matters more than sheer volume.

### T4 — Larger foundation model (3B, LoRA r=64, 30k, 400 steps)

**Purpose:**
Evaluate whether a larger base model (3B parameters) improves task performance and instruction following, and consider the trade-offs for CPU-only inference in the UI.

**Setup:**

- Run name: `model2_3B_r64_30k_400steps`
- Base model: `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`
- Dataset: `train[:30000]`
- LoRA: `r = 64`, `lora_alpha = 64`
- `max_steps = 400` (reduced to keep runtime manageable)

**Metrics:**

- Final training loss: **0.7688**
- Runtime: **3515.17 s** (~58.6 min)
- Peak GPU memory: **4.43 GB**
- LoRA-specific memory: **1.36 GB**

Compared to the best 1B run (T2):

| Metric        | T2 (1B, 10k, 600) | T4 (3B, 30k, 400) |
| ------------- | ----------------- | ----------------- |
| Final loss    | 0.8806            | **0.7688** ⬇⬇     |
| Runtime (s)   | 2649.69           | **3515.17** ⬆     |
| Peak GPU (GB) | 2.955             | **4.43** ⬆        |
| LoRA GPU (GB) | 1.752             | **1.36**          |

**Qualitative evaluation:**

- Explanations (VPN, supervised vs unsupervised, overfitting, Ada Lovelace) are **clear and concise**, often a bit better structured than T2/T3.
- Email and study tips are **high quality** and very usable.
- **System / meta-prompt leakage still appears** (ChatML header).
- **Swedish translation improves slightly but is still wrong**:

  > `Kafféet är för varm att dricka.`
  > (wrong spelling and adjective agreement).

- For the Python function prompt, the model produces a step-by-step explanation rather than just returning a clean function; this is **less aligned** with the instruction than the 1B runs.

**Conclusion (larger model):**

- The 3B model achieves the **lowest training loss** and generally strong natural-language answers.
- However, it **does not fix all alignment issues** (system leakage, translation, strict instruction following for code).
- It is also **more expensive** to train and will be **slower for CPU inference** in the UI. For a lightweight, responsive demo app, the **1B r=64 model (T2)** may remain the more practical choice.

---

## Task 2 Summary & Conclusions

- **Model-centric tuning works best:**
  Increasing LoRA rank from 16 → 64 (T2) provides the most consistent improvement for the 1B model, with only modest cost in VRAM and runtime.

- **Data-centric scaling (more of the same data) has limited impact:**
  Tripling the dataset size from 10k → 30k (T3) does not noticeably improve qualitative behavior.

- **Second foundation model (3B) improves loss but not alignment:**
  The 3B model (T4) trains to a much lower loss and gives strong explanations, but still suffers from occasionally verbose or misaligned answers. Considering the improvement between the 1B and 3B models, it appears that the more advanced models must be using even more features.

- **Best trade-off so far for a CPU-backed UI:**
  Considering quality vs. cost, the **1B, r=64 model (T2)** is a good compromise:

  - Better than T1 and T0 on most tasks
  - More efficient than the 3B model for deployment on CPU-only infrastructure (Hugging Face Spaces / Streamlit Cloud).

Overall, Task 2 demonstrates that **model-centric changes (LoRA rank, base model size)** can significantly affect performance, while naive data scaling has limited benefits without more carefully curated or better-aligned instruction data.