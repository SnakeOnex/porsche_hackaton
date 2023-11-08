import torch
import os
from tqdm import tqdm
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import random_split




class SteeringAngleDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images and steering angles.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.jpg')]
        self.image_paths.sort()  # Ensure that the dataset is sorted by filename

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_paths[idx]
        image = Image.open(img_name)
        angle_name = img_name.replace('.jpg', '.txt')
        
        with open(angle_name, 'r') as file:
            steering_angle = float(file.read().strip())

        sample = {'image': image, 'steering_angle': steering_angle}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

def train(train_dataloader, valid_dataloader, model, criterion, optimizer, num_epochs):
    best_loss = 1e10
    best_model = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()  # Set the model to training mode
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Train'):
            inputs = batch['image'].to(device)
            targets = batch['steering_angle'].to(device).view(-1, 1)  # Ensure the targets are the correct shape

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # No gradients need to be calculated
            for batch in tqdm(validation_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validate'):
                inputs = batch['image'].to(device)
                targets = batch['steering_angle'].to(device).view(-1, 1)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)

        val_loss = val_loss / len(validation_loader.dataset)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')


if __name__ == "__main__":
    # 0. set seed & parameters
    torch.manual_seed(0)
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001

    # 1. setting up dataset and dataloaders
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to the size expected by the conv-net
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize for pre-trained models
    ])
    dataset = SteeringAngleDataset(data_dir='data/', transform=transform)

    ## Define the proportions or absolute numbers for train and validation sets.
    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size

    ## Randomly split the dataset into training and validation datasets.
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    ## setting up dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    # 2. setting up model & training parameters
    model = ...
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train(model, train_dataloader, valid_dataloader, criterion, optimizer, num_epochs)

