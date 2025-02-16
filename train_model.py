import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(file_path, tokenizer, block_size=128):
    """
    Create a TextDataset instance using the provided text data.
    """
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )
    return dataset

def train_model(data_file, output_dir='./trained_model', epochs=5, batch_size=4):
    """
    Fine-tunes GPT-2 on text data collected from multiple open source websites. ok?
    The fine-tuned model is saved in the output directory.
    """
    model_name = "gpt2"
    logger.info("Loading tokenizer and model")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    logger.info(f"Loading dataset from {data_file}")
    dataset = load_dataset(data_file, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=1000,
        save_total_limit=3,
        logging_steps=200,
        learning_rate=5e-5,
        weight_decay=0.01,
    )

    logger.info("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model trained and saved to {output_dir}")

if __name__ == "__main__":
    if os.path.exists("data.txt"):
        train_model("data.txt")
    else:
        print("data.txt not found. Run main.py first to generate data.txt.")
