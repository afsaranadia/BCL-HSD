import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer, AdamW
from huggingface_hub import hf_hub_download
from params import learning_rate, num_classes, dropout_rate, device


class BertCNNBiLSTM(nn.Module):
  def __init__(self, bert_model_name, num_classes):
    super(BertCNNBiLSTM, self).__init__()
    self.bert = AutoModel.from_pretrained(bert_model_name)
    self.cnn = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=3, padding=1)
    self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, bidirectional=True, batch_first=True)
    self.fc = nn.Linear(64 * 2, num_classes)
    self.dropout = nn.Dropout(dropout_rate)

  def forward(self, input_ids, attention_mask):
    bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states = bert_output.last_hidden_state
    cnn_output = self.cnn(hidden_states.transpose(1, 2))
    lstm_output, _ = self.lstm(cnn_output.transpose(1, 2))
    pooled_output = lstm_output[:, -1, :]
    output = self.fc(self.dropout(pooled_output))
    return output



def load_model():
  # Load the model structure
  bert_model_name = "csebuetnlp/banglabert"
  model = BertCNNBiLSTM(bert_model_name=bert_model_name, num_classes=num_classes)

  criterion = nn.CrossEntropyLoss()
  optimizer = AdamW(model.parameters(), lr=learning_rate)

  # Download the model weights from Hugging Face Hub
  weights_path = hf_hub_download(repo_id="nadiaafsara/BCL-HSD-73k", filename="pytorch_model.bin")
  # Load the model weights into the model
  model.load_state_dict(torch.load(weights_path, map_location=device))

  # Move the model to the appropriate device (CPU or GPU)
  model.to(device)

  # Load the tokenizer for your model from Hugging Face Hub
  tokenizer = AutoTokenizer.from_pretrained("nadiaafsara/BCL-HSD-73k")
  print(model)
  return model, tokenizer, criterion, optimizer

def load_model_bhb():
  tokenizer = AutoTokenizer.from_pretrained("saroarj/BanglaHateBert") 
  model = AutoModelForSequenceClassification.from_pretrained("saroarj/BanglaHateBert")
  criterion = nn.CrossEntropyLoss()
  optimizer = AdamW(model.parameters(), lr=learning_rate)
  model.to(device)
  print(model)
  return model, tokenizer, criterion, optimizer

