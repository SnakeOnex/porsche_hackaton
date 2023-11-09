import torch
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import argparse

from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torch.utils.data import random_split

from utils import plot_sample


class SteeringAngleDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        steering_angle = self.steering_angles[idx]

        # load as RGB
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image).float()

        steering_angle = torch.tensor(steering_angle).float()

        # return image, steering_angle
        return {'image': image, 'steering_angle': steering_angle}

    def load_kaggle_dataset(self, data_dir):
        # A. get image paths
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / 'Images'
        self.image_paths = [str(path) for path in self.image_dir.glob('*.png')]
        self.image_paths.sort()  # Ensure that the dataset is sorted by filename

        # B. get steering angles
        self.angles_file_path = self.data_dir / 'SteerValues' / 'steer_values.txt'
        with open(self.angles_file_path, 'r') as file:
            self.steering_angles = [float(angle)
                                    for angle in file.read().splitlines()]

        assert len(self.image_paths) == len(self.steering_angles)
        print(
            f'Loaded Kaggle dataset: {len(self.image_paths)} images with {len(self.steering_angles)} steering angles')

    def print_stats(self):
        steering_angles = np.array(self.steering_angles)
        print(f'Steering angle stats: mean={np.rad2deg(steering_angles.mean()):.4f}, std={np.rad2deg(steering_angles.std()):.4f}, min={np.rad2deg(steering_angles.min()):.4f}, max={np.rad2deg(steering_angles.max()):.4f}')


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = ConvBlock(3, 24, 5, 2, 0)
        self.conv2 = ConvBlock(24, 36, 5, 1, 0)
        self.conv3 = ConvBlock(36, 48, 5, 1, 0)
        self.conv4 = ConvBlock(48, 64, 3, 1, 0)
        self.conv5 = ConvBlock(64, 64, 3, 1, 0)

        self.fc1 = torch.nn.Linear(64, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, x):
        # conv layers
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = F.max_pool2d(self.conv3(x), 2)
        x = F.max_pool2d(self.conv4(x), 2)
        x = F.max_pool2d(self.conv5(x), 2)
        # flatten into a vector
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        # fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(model, train_dataloader, valid_dataloader, criterion, optimizer, num_epochs, device="cpu"):
    best_loss = 1e10
    best_model = None
    for epoch in range(num_epochs):
        # Training phase
        model.train()  # Set the model to training mode
        train_loss = 0.0
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} - Train'):
            inputs = batch['image'].to(device)
            # Ensure the targets are the correct shape
            targets = batch['steering_angle'].to(device).view(-1, 1)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_dataloader.dataset)

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # No gradients need to be calculated
            for batch in tqdm(valid_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} - Validate'):
                inputs = batch['image'].to(device)
                targets = batch['steering_angle'].to(device).view(-1, 1)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)

        val_loss = val_loss / len(valid_dataloader.dataset)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model

            # save model
            torch.save(best_model.state_dict(), f'best_model.pth')

        print(f'Epoch {epoch+1}/{num_epochs}, Train: {train_loss:.4f}, Valid: {val_loss:.4f}, Deg: {np.rad2deg(np.sqrt(val_loss)):.1f}')


def main(args):
    # 0. set seed & parameters
    torch.manual_seed(0)
    max_dataset_size = args.max_dataset_size
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    train_fraction = args.train_frac
    learning_rate = args.lr

    # 1. setting up dataset and dataloaders
    transform = transforms.Compose([
        # Resize the image to the size expected by the conv-net
        transforms.Resize((220, 220)),
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize for pre-trained models
    ])
    dataset = SteeringAngleDataset(transform=transform)
    dataset.load_kaggle_dataset('Data/')
    dataset.print_stats()
    dataset = Subset(dataset, range(max_dataset_size))

    # Define the proportions or absolute numbers for train and validation sets.
    train_size = int(train_fraction * len(dataset))

    # Randomly split the dataset into training and validation datasets.
    # train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    train_dataset = Subset(dataset, range(train_size))
    validation_dataset = Subset(dataset, range(train_size, len(dataset)))

    # setting up dataloader
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available(
    ) else "mps" if torch.backends.mps.is_available() else 'cpu')

    # 2. setting up model & training parameters
    model = ConvNet()
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if not Path("best_model.pth").exists() or args.force_retrain:
        # 3. training
        train(model, train_dataloader, valid_dataloader,
              criterion, optimizer, num_epochs, device=device)

    state_dict = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(state_dict)
    val_idx = 285
    real_idx = train_size + val_idx
    image = dataset[real_idx]['image']
    image_tensor = image.to(device)
    pred = model(image_tensor.unsqueeze(0)).detach().cpu().item()
    steering_angle = dataset[real_idx]['steering_angle']
    plot_sample(image, steering_angle, pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_dataset_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_frac', type=float, default=0.8)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--force_retrain', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
