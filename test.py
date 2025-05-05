import os
import re
import praw
from dotenv import load_dotenv
load_dotenv()
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.optim import AdamW
import csv
from datasets import load_dataset
from datetime import datetime
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

# Create directories
os.makedirs("Processed_Data", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Global constants
MAX_LENGTH = 512
SUMMARY_MAX_LENGTH = 128
BATCH_SIZE = 8
BUFFER_SIZE = 1000
PROCESSED_DATA_PATH = "Processed_Data/processed_tifu.csv"

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Reddit API setup
reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    reddi_username=os.getenv("REDDIT_USERNAME"),
    password=os.getenv("PASSWORD"),
    user_agent=os.getenv("USER_AGENT"),
)

print(os.getenv("CLIENT_ID"))
print(os.getenv("CLIENT_SECRET"))
print(os.getenv("USERNAME"))
print(os.getenv("PASSWORD"))
print(os.getenv("USER_AGENT"))


def extract_comments_from_thread(url, score_threshold=5, limit=100):
    submission = reddit.submission(url=url)
    print(f"\n Extracting from: {submission.title}")

    submission.comments.replace_more(limit=None)
    comments_list = submission.comments.list()

    comments_data = []

    for comment in comments_list[:limit]:
        comment_body = comment.body.replace("\n", " ").strip()
        comment_score = comment.score
        if comment_score < score_threshold:
            continue

        comments_data.append({
            "comment_body": comment_body,
            "comment_score": comment_score
        })

    return comments_data


def load_tifu_data():
    print("Loading reddit_tifu dataset...")

    # Load the dataset
    dataset = load_dataset('reddit_tifu', 'short', trust_remote_code=True)

    # Extract the "short" split which includes TL;DRs
    df = pd.DataFrame(dataset["train"])

    # Filter for posts that have TL;DR summaries
    df = df[df['tldr'].notnull() & (df['tldr'].str.strip() != '')]

    # Rename columns to match our existing pipeline
    df = df.rename(columns={
        'documents': 'comment_body',
        'tldr': 'summary',
        'score': 'comment_score'
    })

    # Add thread_id from the title or ups if available
    if 'ups' in df.columns:
        df['thread_id'] = df['ups']
    else:
        df['thread_id'] = range(len(df))

    print(f"Loaded {len(df)} posts with summaries")

    # Save raw data
    os.makedirs("Raw_Data", exist_ok=True)
    output_file = "Raw_Data/tifu_posts_with_tldr.csv"
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

    return df



def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove Reddit formatting
    text = re.sub(r'\[.*?\]', '', text)

    # Remove TL;DR markers
    text = re.sub(r'TL;DR|TLDR|tl;dr|tldr', '', text, flags=re.IGNORECASE)

    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Remove HTML entities
    text = re.sub(r'&[a-z]+;', '', text)

    # Simplify whitespace
    text = ' '.join(text.split())

    return text.strip()


def prepare_dataset(model_type="t5"):
    print("\nStarting data cleaning and preparation...")

    # Paths
    data_folder = 'Raw_Data'
    file_name = 'tifu_posts_with_tldr.csv'
    file_path = os.path.join(data_folder, file_name)

    processed_data_folder = 'Processed_Data'
    processed_file_path = os.path.join(processed_data_folder, 'processed_tifu.csv')
    os.makedirs(processed_data_folder, exist_ok=True)

    # Load and clean data
    data = pd.read_csv(file_path)
    print(f"Loaded {len(data)} posts")

    # Drop rows with missing or empty summaries
    data = data[data['summary'].notnull() & (data['summary'].str.strip() != '')]
    print(f"After filtering for valid summaries: {len(data)} posts")

    # Clean text
    data['cleaned_text'] = data['comment_body'].apply(clean_text)
    data['cleaned_summary'] = data['summary'].apply(clean_text)

    # Filter: Remove too short texts (after cleaning)
    data = data[
        (data['cleaned_text'].str.len() > 10) &
        (data['cleaned_summary'].str.len() > 5)
        ]
    print(f"After cleaning and filtering short posts: {len(data)} valid posts")


    if model_type == "t5":
        data['formatted_text'] = "summarize: " + data['cleaned_text']
    else:
        data['formatted_text'] = data['cleaned_text']

    # Save processed data
    data_to_save = data[['formatted_text', 'cleaned_summary', 'comment_score']]
    data_to_save.to_csv(processed_file_path, index=False)

    print(f"Processed data saved to: {processed_file_path}")
    return data_to_save




class RedditSummaryDataset(Dataset):
    def __init__(self, texts, summaries, tokenizer, model_type="t5", max_length=MAX_LENGTH,
                 summary_max_length=SUMMARY_MAX_LENGTH):
        self.tokenizer = tokenizer
        self.texts = texts
        self.summaries = summaries
        self.model_type = model_type
        self.max_length = max_length
        self.summary_max_length = summary_max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]

        if self.model_type.startswith("gpt2"):
            # For GPT2, combine text and summary with separator
            combined = text + self.tokenizer.eos_token + summary
            encodings = self.tokenizer(
                combined,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = encodings["input_ids"].squeeze()
            attention_mask = encodings["attention_mask"].squeeze()

            # For GPT2, labels are the same as inputs for causal LM training
            labels = input_ids.clone()
            # Set padding token labels to -100 so they're ignored in loss calculation
            labels[labels == self.tokenizer.pad_token_id] = -100

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        else:
            # For T5 and BART models
            text_encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            with self.tokenizer.as_target_tokenizer():
                target_encoding = self.tokenizer(
                    summary,
                    max_length=self.summary_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

            input_ids = text_encoding["input_ids"].squeeze()
            attention_mask = text_encoding["attention_mask"].squeeze()
            labels = target_encoding["input_ids"].squeeze()

            # Replace padding token id with -100 so it's ignored in loss calculation
            labels[labels == self.tokenizer.pad_token_id] = -100

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }


def prepare_pytorch_datasets(df, tokenizer, model_type="t5", batch_size=BATCH_SIZE):
    print("\nPreparing PyTorch datasets and dataloaders...")

    # Split into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Train set: {len(train_df)} examples, Validation set: {len(val_df)} examples")

    # Create datasets
    train_dataset = RedditSummaryDataset(
        train_df['formatted_text'].tolist(),
        train_df['cleaned_summary'].tolist(),
        tokenizer,
        model_type=model_type
    )

    val_dataset = RedditSummaryDataset(
        val_df['formatted_text'].tolist(),
        val_df['cleaned_summary'].tolist(),
        tokenizer,
        model_type=model_type
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    print(
        f"Created dataloaders with {len(train_dataloader)} training batches and {len(val_dataloader)} validation batches")
    return train_dataloader, val_dataloader



def get_model_and_tokenizer(model_type="t5-small"):
    print(f"\nLoading pretrained {model_type} model and tokenizer...")

    if model_type.startswith("t5"):
        tokenizer = T5Tokenizer.from_pretrained(model_type)
        model = T5ForConditionalGeneration.from_pretrained(model_type)
    elif model_type.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(model_type)
        model = GPT2LMHeadModel.from_pretrained(model_type)
        # GPT2 doesn't have a padding token by default
        tokenizer.pad_token = tokenizer.eos_token
    elif "bart" in model_type:
        tokenizer = BartTokenizer.from_pretrained(model_type)
        model = BartForConditionalGeneration.from_pretrained(model_type)
    else:
        # Default to using T5-small
        print(f"Model type {model_type} not specifically handled, using T5-small")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")

    print(f"Successfully loaded {model_type}")
    return model, tokenizer


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(dataloader)


def fine_tune_model(model, train_dataloader, val_dataloader, model_type="t5", epochs=2):
    print(f"\nFine-tuning {model_type} model for {epochs} epochs...")

    # Move model to device
    model = model.to(device)

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=3e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Training loop
    best_val_loss = float('inf')
    training_losses = []
    validation_losses = []

    # Setup checkpoint directory
    checkpoint_dir = os.path.join("checkpoints", model_type)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        training_losses.append(train_loss)
        print(f"Training loss: {train_loss:.4f}")

        # Evaluate
        val_loss = evaluate(model, val_dataloader, device)
        validation_losses.append(val_loss)
        print(f"Validation loss: {val_loss:.4f}")

        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss: {val_loss:.4f}")

            # Save the model
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pt"))
            print(f"Model saved to {os.path.join(checkpoint_dir, 'best_model.pt')}")

        # Save epoch checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt"))

    # Save final model
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "final_model.pt"))

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.title(f'{model_type} Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, "training_history.png"))
    plt.show()

    print(f"Model fine-tuning complete and saved to {checkpoint_dir}")
    return model, {"train_loss": training_losses, "val_loss": validation_losses}



def generate_summary(text, model, tokenizer, model_type="t5", max_length=350):
    model = model.to(device)
    model.eval()

    cleaned_text = clean_text(text)

    if model_type == "t5":
        input_text = "summarize: " + cleaned_text
    else:
        input_text = cleaned_text

    # Tokenize input
    inputs = tokenizer(
        input_text,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Set model to evaluation mode
    model.eval()

    # Generate summary
    with torch.no_grad():
        if model_type.startswith("gpt2"):
            input_length = inputs["input_ids"].shape[1]
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=input_length + max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                no_repeat_ngram_size=2
            )

            summary = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        else:
            # For T5 and BART models
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )

            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summary





def load_fine_tuned_model(model_type="t5-small", checkpoint_dir=None):
    model, tokenizer = get_model_and_tokenizer(model_type)

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join("checkpoints", model_type)

    # Check if checkpoint exists
    model_path = os.path.join(checkpoint_dir, "best_model.pt")
    if os.path.exists(model_path):
        print(f"Loading fine-tuned model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"No fine-tuned checkpoint found at {model_path}")

    return model, tokenizer



def main():
    print("Starting Reddit TIFU Summarizer with PyTorch")

    # Choose model type
    print("\nChoose a pretrained model to use:")
    print("1. T5-small (Smaller, faster)")
    print("2. BART-base (Good for summarization)")

    choice = input("Enter your choice (1-4): ").strip()

    model_mapping = {
        "1": "t5-small",
        "2": "facebook/bart-base",
    }

    model_type = model_mapping.get(choice, "t5-small")
    print(f"Selected model: {model_type}")

    # Extract the base model type (t5, bart)
    base_model_type = "t5"
    if "bart" in model_type:
        base_model_type = "bart"
    elif "gpt2" in model_type:
        base_model_type = "gpt2"

    # Part 1: Load dataset (if not already loaded)
    if not os.path.exists("Raw_Data/tifu_posts_with_tldr.csv"):
        try:
            load_tifu_data()
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return
    else:
        print("Raw dataset file already exists.")

    # Part 2: Clean and prepare data for the specific model
    try:
        prepare_dataset(model_type=base_model_type)
    except Exception as e:
        print(f"Error in data preparation: {e}")
        return

    # Load processed data
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print(f"Loaded {len(df)} processed examples")

    # Part 3: Load model and tokenizer
    try:
        model, tokenizer = get_model_and_tokenizer(model_type)
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        return

    # Check if we should load a fine-tuned model or train a new one
    use_pretrained = input("\nDo you want to load a previously fine-tuned model? (yes/no): ").strip().lower()

    if use_pretrained == 'yes':
        try:
            model, tokenizer = load_fine_tuned_model(model_type)
        except Exception as e:
            print(f"Error loading fine-tuned model: {e}")
            print("Proceeding to train a new model...")
            use_pretrained = 'no'

    if use_pretrained != 'yes':
        # Prepare dataset for PyTorch
        train_dataloader, val_dataloader = prepare_pytorch_datasets(
            df, tokenizer, model_type=base_model_type, batch_size=BATCH_SIZE)

        # Fine-tune the model
        epochs = int(input("\nEnter number of epochs for fine-tuning (recommended 2-5): ") or "3")
        try:
            model, history = fine_tune_model(
                model, train_dataloader, val_dataloader,
                model_type=base_model_type,
                epochs=epochs
            )
        except Exception as e:
            print(f"Error during model fine-tuning: {e}")
            return

    # Part 4: Test the model

    async def  interactive_testing(link):
            # Handle Reddit thread input
            if await link.startswith("http"):
                try:
                    comments = extract_comments_from_thread(link)
                    if not comments:
                        print("No comments above the score threshold.")
                          

                    full_text = " ".join([c["comment_body"] for c in comments])

                    # ensure better summarization if not enough data is collected from the thread.
                    if len(full_text.strip()) < 10:
                        print("Not enough content to summarize.")
                       

                    summary = generate_summary(
                        full_text, model, tokenizer, model_type=base_model_type
                    )
                    print(f"\nCombined Summary for Thread:\n{summary}")

                except Exception as e:
                    print(f"Error: {e}. Ensure the URL is a valid Reddit thread.")

            # Handle manual text input
            else:
                if len(user_input.strip()) < 10:
                    print("Please enter a longer text for better results.")
                  

                summary = generate_summary(
                    user_input, model, tokenizer, model_type=base_model_type
                )
                print(f"Generated Summary: {summary}")

    interactive_testing(link)

    print(f"\nReddit TIFU Summarizer with PyTorch {model_type} Completed!")


if __name__ == "__main__":
    main()