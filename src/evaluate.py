import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from params import learning_rate, epochs, batch_size, patience, device
from model import load_model
from data_loader import load_data
import matplotlib.pyplot as plt
import seaborn as sns

model, tokenizer, criterion, optimizer = load_model()

def evaluate_model(model, test_loader):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="weighted")
    precision = precision_score(true_labels, predictions, average="weighted")
    recall = recall_score(true_labels, predictions, average="weighted")

    # Generate confusion matrix for test set
    conf_matrix = confusion_matrix(true_labels, predictions)
    # Generate classification report
    report = classification_report(true_labels, predictions, target_names=['Non-Hate', 'Hate'])

    print(f"Accuracy: {acc}, F1: {f1}, Precision: {precision}, Recall: {recall}")
    return conf_matrix, report


if __name__ == "__main__":
    # Evaluate the model
    conf_matrix, report = evaluate_model(model, test_loader)

    # Plot confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Hate', 'Hate'], yticklabels=['Non-Hate', 'Hate'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Test Set')
    plt.savefig(f"../outputs/plots/confusion_matrix.png", bbox_inches="tight")
    plt.show()
    # Print classification report
    print(report)
    # Save the classification report as a text file
    with open("../outputs/plots/classification_report.txt", "w") as f:
      f.write(report)