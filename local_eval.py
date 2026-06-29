import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

# Configuration
MAX_LENGTH = 150
SAVED_MODEL_PATH = "../saved_models"
TEST_DATA_PATH = "../data/test_data.csv"
BATCH_SIZE = 32  # Adjust based on your memory
DEVICE = "cpu"

# Set number of threads for PyTorch
torch.set_num_threads(8)

# Load the model and tokenizer
config = AutoConfig.from_pretrained(SAVED_MODEL_PATH, trust_remote_code=True)
config.problem_type = "multi_label_classification"
config.num_labels = MAX_LENGTH

model = AutoModelForSequenceClassification.from_pretrained(SAVED_MODEL_PATH, config=config, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(SAVED_MODEL_PATH, trust_remote_code=True)

#print("Model Configuration:", model.config)

# Load and preprocess test data
test_df = pd.read_csv(TEST_DATA_PATH)

def process_labels(labels):
    return [int(x) for x in labels.split(',')]

test_df['label'] = test_df['label'].apply(process_labels)

test_encodings = tokenizer(
    list(test_df['sequence']),
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt"
)

# Convert labels to tensor
test_labels = torch.tensor(test_df['label'].tolist(), dtype=torch.float32)

# Create a custom dataset
class DNADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

test_dataset = DNADataset(test_encodings, test_labels)

# Create DataLoader
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Evaluation function
def evaluate_model(model, data_loader, device="cpu"):
    model.eval()  # Set model to evaluation mode
    all_logits = []
    all_labels = []

    with torch.no_grad():  # No need for gradients during evaluation
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Collect logits and labels
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all logits and labels
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Apply sigmoid to convert logits to probabilities
    probs = torch.sigmoid(all_logits)

    # Convert probabilities to binary predictions
    preds = (probs > 0.5).numpy()
    all_labels = all_labels.numpy()

    return preds, all_labels

# Run evaluation
predictions, true_labels = evaluate_model(model, test_loader, device=DEVICE)

# Compute metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average="samples")
recall = recall_score(true_labels, predictions, average="samples")
f1 = f1_score(true_labels, predictions, average="samples")

print("Evaluation Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Classification report
print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=[f"Label_{i}" for i in range(true_labels.shape[1])]))
