import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import subprocess
import time
from threading import Thread

# def log_gpu_utilization(log_file, interval=300):
#     """Logs GPU utilization to a file at regular intervals."""
#     with open(log_file, "w") as f:
#         f.write("Time, GPU Utilization (%), Memory Usage (MB)\n")  # Header
#         while True:
#             try:
#                 # Run the nvidia-smi command
#                 result = subprocess.run(
#                     ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
#                     stdout=subprocess.PIPE,
#                     stderr=subprocess.PIPE,
#                     text=True,
#                 )
#                 if result.returncode == 0:
#                     timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#                     gpu_data = result.stdout.strip().replace("\n", " | ")
#                     f.write(f"{timestamp}, {gpu_data}\n")
#                     f.flush()
#                 else:
#                     f.write(f"Error: {result.stderr}\n")
#             except Exception as e:
#                 f.write(f"Logging failed: {e}\n")
#             time.sleep(interval)


# # Start GPU logging in a separate thread
# log_file = "log_directory"
# gpu_logger_thread = Thread(target=log_gpu_utilization, args=(log_file, 300), daemon=True)
# gpu_logger_thread.start()

# Adjust sequence length (MAX_LENGTH)
MAX_LENGTH = 150

model_path = "model_path"
tokenizer_path = "tokenizer_path"

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
config.problem_type = "multi_label_classification"
config.num_labels = MAX_LENGTH

model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

train_df = pd.read_csv('../data/train_data.csv')
test_df = pd.read_csv('../data/test_data.csv')
dev_df = pd.read_csv('../data/dev_data.csv')


def tokenize_function(examples):
    return tokenizer(examples['sequence'], padding='max_length', truncation=True, max_length=MAX_LENGTH)


def process_labels(labels):
    return list(map(int, labels.split(',')))


train_df['label'] = train_df['label'].apply(process_labels)
dev_df['label'] = dev_df['label'].apply(process_labels)
test_df['label'] = test_df['label'].apply(process_labels)

# Tokenize the data
train_encodings = tokenizer(list(train_df['sequence']), padding=True, truncation=True, max_length=MAX_LENGTH)
dev_encodings = tokenizer(list(dev_df['sequence']), padding=True, truncation=True, max_length=MAX_LENGTH)
test_encodings = tokenizer(list(test_df['sequence']), padding=True, truncation=True, max_length=MAX_LENGTH)

# Convert labels to tensors
train_labels = torch.tensor(train_df['label'].tolist(), dtype=torch.float32)
dev_labels = torch.tensor(dev_df['label'].tolist(), dtype=torch.float32)
test_labels = torch.tensor(test_df['label'].tolist(), dtype=torch.float32)


class DNADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = DNADataset(train_encodings, train_labels)
dev_dataset = DNADataset(dev_encodings, dev_labels)
test_dataset = DNADataset(test_encodings, test_labels)

# Training arguments
training_args = TrainingArguments(
    output_dir='output_directory',          # output directory
    evaluation_strategy="steps",     # evaluation strategy
    learning_rate=3e-5,              # learning rate
    per_device_train_batch_size=12,   # reduce batch size to save memory
    per_device_eval_batch_size=32,    # reduce evaluation batch size
    gradient_accumulation_steps=2,   # accumulate gradients to maintain effective batch size
    num_train_epochs=3,              # number of training epochs
    weight_decay=0.01,               # strength of weight decay
    fp16=True,                       # enable mixed precision training
    logging_dir='logging_directory',            # directory for storing logs
    logging_steps=500,                # log every 10 steps
    save_steps=1000,                  # save model every 200 steps
    overwrite_output_dir=False       # overwrite output directory
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer
)

trainer.train(resume_from_checkpoint=True)

results = trainer.evaluate(test_dataset)
print(results)

trainer.save_model("save_location")
tokenizer.save_pretrained("save_location")