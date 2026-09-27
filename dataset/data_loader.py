from torch.utils.data import DataLoader
from prepare_data import train_dataset, val_dataset

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
