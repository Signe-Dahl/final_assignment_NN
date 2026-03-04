# Final Assignment - Green Patent Detection: Advanced Agentic Workflow with QLoRA
Video: https://aaudk-my.sharepoint.com/:v:/g/personal/de63sv_student_aau_dk/IQCqlEBC1qfjTavcjynR_k3cAf_-9tW2JsVn2i2KgLv6_9U?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D&e=tdeFi1

Model: https://huggingface.co/Signe22/patentsberta-green-hitl-final

Dataset: https://huggingface.co/datasets/Signe22/patents-50k-green-hitl-final

## QLORA fine-tuning
The fine-tuning was done in AI-Lab, the code is displayed below (for reference only, NOT to be run).

### QLoRA Fine-Tuning of Mistral-7B
To specialize a large language model for green patent classification, I fine-tuned Mistral-7B-Instruct-v0.2 using QLoRA (Quantized Low-Rank Adaptation). This approach enables efficient domain adaptation while significantly reducing GPU memory requirements.

Noteable QLoRA parameters:
- Rank (r): 16
    - The rank determines the size of the low-rank update matrices inserted into the transformer layers. A higher rank increases the adapter’s capacity to learn task-specific patterns but also increases the number of trainable parameters.
    - 16 was chosen as a balanced trade off between adaptation capacity and parameter efficiency, providing sufficient task-specific learning ability.
- LoRA alpha: 32
    - Scaling factor applied to the LoRA updates. It controls the strength of the adapter’s contribution relative to the frozen base model.
    - Setting alpha to 32 (2 * rank) ensures the adapter has meaningful influence on the model’s output while keeping updates stable.
- LoRA dropout: 0.05
    - Dropout rate applied to the LoRA layers during training to improve generalization and reduce overfitting.

QLoRA allows training only a small set of injected low-rank matrices while keeping the base model frozen in 4-bit precision. This reduces resource requirements while maintaining strong adaptation capability.

Training was performed using the train_silver split from patents_50k_green.parquet.

Training Configuration:
- Learning rate: 2e-5
    - A relatively small learning rate is used to ensure stable adaptation of the LoRA layers without destabilizing the pretrained base model.
- Epochs: 1
    - The model is trained for a single epoch, meaning it sees the entire training dataset once. This choice was made due to computational constraints and to reduce training time while still allowing the model to adapt to the task.

Training loss decreased steadily and stabilized, indicating successful convergence of the LoRA adapter.

Only the LoRA adapter weights were saved, meaning I have created a lightweight domain-adapted classifier while keeping the base model unchanged.

## Agentic classification setup
This time I chose to do the classification using agentic setup in AI-Lab as well due to computational resource needs.

The script shown below implements a green patent classification pipeline using a QLoRA-adapted Mistral model within an agentic setup.

The base model (Mistral-7B-Instruct-v0.2) is loaded in 4-bit quantized form using bitsandbytes to reduce memory usage. A LoRA adapter trained on green patent classification is then loaded using:

judge_model = PeftModel.from_pretrained(base_model, ADAPTER_LOCAL)

Only the Judge agent uses the QLoRA adapter. The Advocate and Skeptic agents use the base quantized model without adapters. This isolates domain specialization to the final decision step.

Intermediate results are checkpointed periodically to prevent data loss.

### Agentic Structure
For each patent claim:
- Advocate Agent (base model): Reads only the claim text and produces a short argument for classifying the claim as green technology, along with a binary stance (0/1) and an argument strength level (weak/medium/strong) in structured JSON.
- Skeptic Agent (base model): Reads only the claim text and produces a short argument against green classification, also returning a binary stance (0/1) and strength (weak/medium/strong) in structured JSON.
- Judge Agent (QLoRA-adapted model): Receives the claim text plus the Advocate and Skeptic arguments and returns the final green label (llm_green_suggested = 0/1), confidence level (low/medium/high), and a one-sentence rationale in structured JSON.

### Output Consistency Handling
Because the agents are generative models, strict JSON output could not be guaranteed by prompting alone. Several safeguards are implemented:
- Schema validation (Pydantic): The output is validated against a strict schema. If invalid, the model retries once before marking the row as failed.
- Normalization: Escaped underscores (e.g., llm\_green_suggested) are corrected before parsing.
- Robust JSON extraction: A brace-scanning function extracts the first valid JSON object containing required fields.

These steps ensure that even if the agents sometimes generate free text, the final stored output is reliable and structured.

## Disagreement Reporting

### Agentic disagreement between the advocate and the skeptic
I had the advocate and skeptic output their stance (their 'vote') for each claim, where they were told to choose between 0 and 1. Out of 100 claims total, the agents disagreed on 60. This is a high level of disagreement, but it is relevant to note that theses patents are specifically chosen due to their uncertainty.

## HITL
I will now load the gold_100 dataset, which was classified by my agentic setup (prior to the HITL step) in this run. For a combined implementation of an agentic setup and HITL within the same pipeline, I refer to my Assignment 3 work using CrewAI and LangChain.

This run uses an exception-based HITL approach, meaning I do not manually review all 100 claims. Instead, I focus human review on:

- cases where there is disagreement between agents.
- cases where the Judge’s confidence is low.

However, the Judge flagged 45 claims as low-confidence (likely due to it being high uncertainty claims), which indicates that substantial human review is still required with the current setup.

## Model Training
The model was fine-tuned for one epoch using a maximum sequence length of 256 tokens and a learning rate of 2e-5, following the recommended settings to keep computation reasonable. Tokenization was performed using the PatentSBERTa tokenizer prior to training.

Model performance was evaluated on the held-out eval_silver split and gold_100 subset so evaluate performance, which will be used for a comparative analysis between baseline model, A2 model, A3 model and this final model.

## Comparative Analysis

### Model Performance Comparison

The table below compares model performance across the four stages of the project using the same silver evaluation set.

| Model Version | Training Data Source | F1 Score (Eval Set) |
|---------------|----------------------|---------------------|
| Baseline | Frozen PatentSBERTa embeddings (no fine-tuning) | 0.7719 |
| Assignment 2 Model | Fine-tuned on Silver + Gold (Simple LLM labels + HITL) | 0.8030 |
| Assignment 3 Model | Fine-tuned on Silver + Gold (Agentic LLM + HITL) | 0.8051 |
| Final Assignment Model | Fine-tuned on Silver + Gold (QLoRA-Powered MAS + Targeted HITL) | 0.8041 |

### Reflection
The final QLoRA-powered Multi-Agent System (MAS) model achieved an F1 score of 0.8041 on the independent evaluation set, which is marginally lower than the Assignment 3 model (0.8051) but higher than earlier versions. The difference (0.001) is small, suggesting that the fine-tuned models perform at a comparable level on unseen data.

Although the final model achieved the highest F1 score on the Gold 100 subset, this dataset was also included during fine-tuning and therefore cannot be considered an independent evaluation benchmark. While the improved performance likely indicates stronger alignment with the high-quality gold annotations, it may also partially reflect closer fitting to examples seen during training. As such, results on this subset should be interpreted as evidence of improved calibration and label alignment rather than definitive proof of improved generalization.

Overall, the results indicate that the QLoRA-powered MAS with targeted HITL improved alignment with human judgments while maintaining stable performance on the broader evaluation set.
