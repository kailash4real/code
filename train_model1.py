from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, TextDataset, DataCollatorForLanguageModeling, Trainer
import os

# Paths
DATA_DIR = "./data/info"
TRAIN_FILE = os.path.join(DATA_DIR, "cord19_abstracts.csv")
MODEL_DIR = "./models/gpt2_finetuned"
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer")
MODEL_PATH = os.path.join(MODEL_DIR, "model")

# Load pretrained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")

# Prepare dataset
dataset = TextDataset(tokenizer=tokenizer, file_path=TRAIN_FILE, block_size=128)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments and Trainer
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Fine-tune the pretrained model on your dataset
trainer.train()
trainer.save_model(MODEL_PATH)
