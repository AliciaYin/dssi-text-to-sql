import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType

# Load config from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    print("CONFIG LOADED:", config)


# Load model and tokenizer
model_id = config["model_name_or_path"]
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

# Apply LoRA configuration
lora = config["peft"]["lora_config"]
lora_config = LoraConfig(
    r=lora["r"],
    lora_alpha=lora["lora_alpha"],
    target_modules=lora["target_modules"],
    lora_dropout=lora["lora_dropout"],
    bias=lora["bias"],
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, lora_config)

# Load dataset (Spider, first N samples)
dataset = load_dataset(config["dataset_name"], split=config["dataset_split"])

# Preprocess function
def preprocess(example):
    inputs = f"Translate to SQL: {example[config['text_column']]}"
    targets = example[config["label_column"]]
    model_inputs = tokenizer(inputs, max_length=config["max_seq_length"], padding="max_length", truncation=True)
    labels = tokenizer(targets, max_length=config["max_seq_length"], padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize dataset
tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=config["output_dir"],
    per_device_train_batch_size=config["per_device_train_batch_size"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    num_train_epochs=config["num_train_epochs"],
    logging_steps=config["logging_steps"],
    save_strategy=config["save_strategy"],
    save_total_limit=config["save_total_limit"],
    learning_rate=config["learning_rate"],
    max_grad_norm=config["max_grad_norm"],
    fp16=config["fp16"]
)

# Train
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

# Save LoRA adapter
model.save_pretrained("lora_codet5_adapter")
