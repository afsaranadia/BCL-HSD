import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from params import learning_rate, epochs, batch_size, patience, device
from model import load_model
from data_loader import load_data
import matplotlib.pyplot as plt
import seaborn as sns

model, tokenizer, criterion, optimizer = load_model()

bcl_acc_train = []
bcl_f1_train = []
bcl_pre_train = []
bcl_rec_train = []
bcl_acc_val = []
bcl_f1_val = []
bcl_pre_val = []
bcl_rec_val = []

class EarlyStopping:
    def __init__(self, patience=patience, mode="max", delta=0.001):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best_metric = None
        self.counter = 0
        self.stop_training = False

    def __call__(self, current_metric):
        if self.best_metric is None or (
            self.mode == "min" and current_metric < self.best_metric - self.delta
        ) or (
            self.mode == "max" and current_metric > self.best_metric + self.delta
        ):
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True


def compute_metrics(predictions, labels):
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")
    return acc, f1, precision, recall


def train_model_with_early_stopping_and_metrics(
    model, train_loader, valid_loader, criterion, optimizer, epochs=epochs, patience=patience
):
    early_stopping = EarlyStopping(patience=patience, mode="max")  # Monitoring validation accuracy
    best_model_state = None
    best_epoch = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_predictions, train_labels = [], []
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Collect predictions and labels
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            train_predictions.extend(preds)
            train_labels.extend(labels.cpu().numpy())

        train_acc, train_f1, train_precision, train_recall = compute_metrics(train_predictions, train_labels)
        bcl_acc_train.append(train_acc )
        bcl_f1_train.append(train_f1)
        bcl_pre_train.append(train_precision)
        bcl_rec_train.append(train_recall)
        print(
            f"Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss / len(train_loader):.4f}, "
            f"Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}"
        )

        # Validation phase
        model.eval()
        valid_predictions, valid_labels = [], []
        valid_loss = 0

        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                # Collect predictions and labels
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                valid_predictions.extend(preds)
                valid_labels.extend(labels.cpu().numpy())

        valid_acc, valid_f1, valid_precision, valid_recall = compute_metrics(valid_predictions, valid_labels)
        bcl_acc_val.append(valid_acc)
        bcl_f1_val.append(valid_f1)
        bcl_pre_val.append(valid_precision)
        bcl_rec_val.append(valid_recall)
        print(
            f"Epoch {epoch + 1}/{epochs}, Validation Loss: {valid_loss / len(valid_loader):.4f}, "
            f"Accuracy: {valid_acc:.4f}, F1: {valid_f1:.4f}, Precision: {valid_precision:.4f}, Recall: {valid_recall:.4f}"
        )

        # Early stopping check
        early_stopping(valid_acc)  # Monitor validation accuracy
        if early_stopping.stop_training:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        # Save the best model state
        if early_stopping.best_metric == valid_acc:
            best_model_state = model.state_dict()
            best_epoch = epoch + 1

    # Restore the best model state
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Loaded the best model from epoch {best_epoch}")

    return model


if __name__ == "__main__":

  train_loader,valid_loader, _ = load_data("hf://datasets/nadiaafsara/Bhs-Kag/BSH_Kag_30k.csv")
  trained_model = train_model_with_early_stopping_and_metrics(
    model, train_loader, valid_loader, criterion, optimizer, epochs=epochs, patience=patience
  )


  # Plot figures
  # Plot training and validation metrics
  epochs_range = range(1, len(bcl_acc_train) + 1)

  # Plot Accuracy
  plt.figure(figsize=(10, 6))
  plt.plot(epochs_range, bcl_acc_val, label="Validation Accuracy")
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.title("Training and Validation Accuracy")
  plt.legend()
  plt.savefig(f"../outputs/plots/training_validation_accuracy.png", bbox_inches="tight")
  plt.show()

  # Plot F1 Score
  plt.plot(epochs_range, bcl_f1_train, label="Train F1 Score")
  plt.xlabel("Epochs")
  plt.ylabel("F1 Score")
  plt.title("Training and Validation F1 Score")
  plt.legend()
  plt.savefig(f"../outputs/plots/training_validation_f1_score.png", bbox_inches="tight")
  plt.show()

  # Plot Precision
  plt.figure(figsize=(10, 6))
  plt.plot(epochs_range, bcl_pre_train, label="Train Precision")
  plt.xlabel("Epochs")
  plt.ylabel("Precision")
  plt.title("Training and Validation Precision")
  plt.legend()
  plt.savefig(f"../outputs/plots/training_validation_precision.png", bbox_inches="tight")
  plt.show()

  # Plot Recall
  plt.figure(figsize=(10, 6))
  plt.plot(epochs_range, bcl_rec_train, label="Train Recall")
  plt.plot(epochs_range, bcl_rec_val, label="Validation Recall")
  plt.xlabel("Epochs")
  plt.ylabel("Recall")
  plt.title("Training and Validation Recall")
  plt.legend()
  plt.savefig(f"../outputs/plots/training_validation_recall.png", bbox_inches="tight")
  