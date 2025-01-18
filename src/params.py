import torch
num_classes = 2 # Binary classification

learning_rate = 2e-5
batch_size = 64
epochs = 10
patience = 3

dropout_rate = 0.3


dataset_73k_path = "hf://datasets/Sky1241/hsban_merge_73k/merged_dataset73k.csv"
dataset_30k_path = "hf://datasets/nadiaafsara/Bhs-Kag/BSH_Kag_30k.csv"

print('Using PyTorch version:', torch.__version__)
if torch.cuda.is_available():
    print('Using GPU, device name:', torch.cuda.get_device_name(0))
    device = torch.device('cuda')
else:
    print('No GPU found, using CPU instead.')
    device = torch.device('cpu')




  