# DSSI Day3 - Text-to-SQL Ollama Model Assessment

## 🔍 Project Overview

This project implements a text-to-SQL model using `CodeT5+ 220M` fine-tuned with LoRA adapters. It allows natural language queries to be converted into SQL statements. The entire system runs on a Mac with 8GB RAM and no GPU.

## 🧠 Model Training

- **Base Model:** `Salesforce/codet5p-220m`
- **Technique:** LoRA (Low-Rank Adaptation)
- **Framework:** Hugging Face Transformers + PEFT
- **Dataset:** Spider (first 1000 samples)
- **Config:** `config.yaml` used for training parameters

## ⚙️ Web App

- **Framework:** Streamlit
- **File:** `app.py`
- Allows users to input natural language and receive SQL queries in real-time.

## 🧪 Ollama Integration

> ⚠️ Ollama does not support Seq2Seq models like CodeT5+ as of now.  
> Deployment simulation is shown via `Modelfile` and documented commands.  
> If support is added in future, `lora_codet5_adapter/` can be integrated accordingly.

## 🚀 How to Run

### Setup

```bash
python -m venv text2sql-venv
source text2sql-venv/bin/activate  # Or .\text2sql-venv\Scripts\activate on Windows
pip install -r requirements.txt

Author
Yin Thwe Thwe
