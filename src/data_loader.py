import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset
import normalization
from params import batch_size
from model import load_model


_, tokenizer,_,_ = load_model()


# Tokenize the dataset
def tokenize_data(samples):
    return tokenizer(
        samples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )



def load_data(data_path):
    # Load the dataset
    df = pd.read_csv(data_path)
    df = df.rename(columns={df.columns[0]: "text", df.columns[1]: "label"})

    
    # Normalize
    df = df.dropna()  # Remove any rows with missing values
    l = len(df.text)
    n = normalization.Normalize()

    for i in range (l):
      text = df.text[i]
      ct = n.normalize_text(text)
      if ct=='':
        df = df.drop([i])#Dropping empty row
      else:
        df.loc[i,'text'] = ct
    
    # Convert dataframe to dataset
    dataset = Dataset.from_pandas(df)

    #Split dataset
    dataset = dataset.train_test_split(test_size=0.2)
    test_data = dataset["test"].train_test_split(test_size=0.5)
    train_data = dataset["train"]
    valid_data = test_data["train"]
    test_data = test_data["test"]

    print(len(train_data))
    print(len(valid_data))
    print(len(test_data))

    # Apply tokenization to the datasets
    train_data = train_data.map(tokenize_data, batched=True)
    valid_data = valid_data.map(tokenize_data, batched=True)
    test_data = test_data.map(tokenize_data, batched=True)

    print(train_data.column_names)
    print(valid_data.column_names)
    print(test_data.column_names)

    # Remove unnecessary columns
    train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    valid_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Define DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, valid_loader, test_loader
    



if __name__ == "__main__":
    load_data("hf://datasets/nadiaafsara/Bhs-Kag/BSH_Kag_30k.csv")
    print("Data prepared successfully.")
