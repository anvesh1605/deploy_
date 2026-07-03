# Aruba QA Training Pipeline - Mentor Notes

## Purpose
This handout captures the talking points for a short mentor presentation about the project build. The main idea is simple:

- `Qwen` is the fine-tuned main answer generator.
- `BiLSTM` is the supporting module for intent routing.
- Deterministic lookup is the factual source of truth.
- The system is grounded first, then Qwen turns the grounded answer into natural language.

## What To Say First
Use this one-sentence summary at the start:

The project is a grounded Aruba QA pipeline where a fine-tuned Qwen model generates answers, a BiLSTM supporting module handles intent routing, and deterministic lookup keeps the answers factual and consistent.

## Slide 1: System Overview
### Speaker Notes
- This project answers Aruba release-note and product-documentation questions locally.
- It is not a free-form chatbot.
- The workflow combines three parts:
  - a fine-tuned Qwen model
  - a BiLSTM supporting module
  - deterministic lookup
- The purpose of the design is to keep factual answers stable while still making the output natural and readable.

### What To Emphasize
- Qwen is the answer generation model.
- BiLSTM is not the main answer model.
- Lookup provides the actual facts.

## Slide 2: How The System Is Built
### Speaker Notes
- A user question enters the system.
- Slot extraction finds structured clues such as switch, version, sub-version, bug ID, feature, category, command, or topic.
- The BiLSTM supporting module predicts the intent.
- Deterministic lookup retrieves the exact answer from the indexed corpus.
- Qwen receives only the grounded answer and rewrites it naturally.
- If validation fails, the pipeline falls back to the lookup answer directly.

### What To Emphasize
- The system is modular.
- Routing and grounding happen before generation.
- Qwen never gets to invent the facts.

## Slide 3: Inference Flow
### Speaker Notes
- Show the flow as:
  - User question
  - Slot extraction
  - BiLSTM intent prediction
  - Deterministic lookup
  - Fine-tuned Qwen grounded generation
  - Validation
  - Final answer
- Explain that the lookup answer is the canonical factual answer.
- Mention that validation protects against paraphrases that change meaning, bug IDs, versions, commands, or workaround text.
- If the output is too risky, the system returns the lookup answer instead.

### What To Emphasize
- Grounded first.
- Generative second.
- Validation is the safety gate.

## Slide 4: Why This Design Works
### Speaker Notes
- BiLSTM gives lightweight intent control and routing.
- Qwen improves readability and conversation quality.
- Deterministic lookup protects factual correctness.
- This split reduces hallucination and keeps answers consistent across release notes and product documentation.
- The approach is practical for local deployment and easier to debug than a single free-form model.

### What To Emphasize
- Reliability over creativity.
- Facts come from lookup.
- Language polish comes from Qwen.

## Important Notes
- Do not describe BiLSTM as the main answer model.
- Do not describe Qwen as the source of facts.
- Say "BiLSTM supporting module" instead of "full support model."
- Say "deterministic lookup" instead of "LLM memory."
- Keep the story focused on architecture, not on dataset preparation.

## Short Closing Line
This project combines a fine-tuned Qwen model with a BiLSTM supporting module and deterministic lookup so the system stays grounded, accurate, and easy to use.

