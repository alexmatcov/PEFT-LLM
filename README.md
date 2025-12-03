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

### Summary

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