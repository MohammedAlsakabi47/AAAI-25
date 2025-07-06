## Imports:
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time



## Prepare Dataset
class MyDataset(Dataset):
    def __init__(self, data_folder, gt_folder, transform=None):
        self.data_folder = data_folder
        self.gt_folder = gt_folder
        self.transform = transform
        self.data_files = [f for f in os.listdir(data_folder) if self.is_image_file(f)]
        self.data_files = sorted(self.data_files, key=lambda x: int(x.split('.')[0])) # Sort files correctly
        self.data_files = self.data_files[:1000]
        self.gt_files = [f for f in os.listdir(gt_folder) if self.is_image_file(f)]
        self.gt_files = sorted(self.gt_files, key=lambda x: int(x.split('.')[0])) # Sort files correctly
        self.gt_files = self.gt_files[:1000]
        self.length = min(len(self.data_files), len(self.gt_files))  # Ensure both folders have same number of files

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        data_path = os.path.join(self.data_folder, self.data_files[idx])
        gt_path = os.path.join(self.gt_folder, self.gt_files[idx])

        # Load images
        data_img = Image.open(data_path).convert('L')  # Ensure data is loaded as RGB
        gt_img = Image.open(gt_path).convert('L')  # Ensure ground truth is loaded as RGB

        # Apply transformations if specified
        if self.transform:
            data_img = self.transform(data_img)
            gt_img = self.transform(gt_img)

        return data_img, gt_img
    
    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in ['.jpg', '.jpeg', '.png', '.bmp', '.gif'])


## Modify ResNet101 to accept single-channel input
def modify_resnet101():
    model = models.resnet101(pretrained=True)

    # Modify the first convolutional layer to accept 1 input channel
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Modify the final layer to output 1 channel (for grayscale image)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 256 * 256)
    return model


if __name__=='__main__':
    # Check if GPU is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Define transformations if needed
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images if necessary
        transforms.ToTensor(),  # Convert PIL Image to PyTorch Tensor
    ])

    # Assuming your dataset and ground truth folders are named accordingly
    data_folder = 'For_testing/radar depth map spectrum'
    gt_folder = 'Example Frames/camera spetrum/'

    # Create dataset instance
    dataset = MyDataset(data_folder, gt_folder, transform=transform)

    # Define hyperparameters
    num_epochs = 10000
    learning_rate = 0.0001
    batch_size = 10

    # Create Subsets for training and testing
    train_indices = list(range(0, n_samples))
    test_indices = list(range(10,n_samples-1,10))
    train_indices = list(set(train_indices) - set(test_indices))
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    
    # Create dataloader instance
    shuffle = False  # Shuffle the data
    dataloader = DataLoader(dataset, batch_size=batch_size_all, shuffle=shuffle)
    train_loader = DataLoader(train_subset, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size_test, shuffle=False)
    
    # Wrap the model with nn.DataParallel
    model = modify_resnet101()
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    # Initialize the model, loss function, and optimizer
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    loss_history = []
    best_loss = 100
    for epoch in range(num_epochs):
        start_time = time.time()  # Record start time of epoch
        for low_res, high_res in train_loader:
            low_res, high_res = low_res.to(device), high_res.to(device)

            # Forward pass
            outputs = model(low_res)
            # print('here: ', np.size(outputs))
            outputs = outputs.view(batch_size, 1, 256, 256)
            loss = criterion(outputs, high_res)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save model
            # if loss <= best_loss:
            #     best_loss = loss
            #     torch.save(model.state_dict(), 'Best Model')

            loss_history.append(loss.item())
        epoch_time = time.time() - start_time  # Calculate epoch time
        if epoch % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Time: {epoch_time}')

    # Save loss
    with open('loss_history.pkl', 'wb') as f:
        pickle.dump(loss_history, f)
    model.eval()
    counter = 1
    with torch.no_grad():
        for low_res, high_res in test_loader:
            low_res, high_res = low_res.to(device), high_res.to(device)
            output = model(low_res)
            output = output.view(batch_size, 1, 256, 256)
            print('Loss Test: ', criterion(output, high_res).item())

            for ii in range(batch_size):            
                output_cpu = output.cpu()
                high_res_cpu = high_res.cpu()
                input_cpu = low_res.cpu()

                plt.figure()
                plt.subplot(1,3,1)
                plt.imshow((input_cpu.detach().numpy())[ii,0], cmap='gray')
                plt.title('Encoded Radar Output')

                plt.subplot(1,3,2)
                plt.imshow((output_cpu.detach().numpy())[ii,0], cmap='gray')
                plt.title('Enhanced Encoded Radar Output')

                plt.subplot(1,3,3)
                plt.imshow((high_res_cpu.detach().numpy())[ii,0], cmap='gray')
                plt.title('Ground Truth')

                # Save the grayscale image as JPEG
                output_path = 'For_testing/Results/'+str(counter)+'.jpg' # Path to save results
                plt.savefig(output_path)
                plt.close()
                
                # Save output as an array
                np.save('For_testing/numpy matrices/'+str(counter)+'.npy', (output_cpu.detach().numpy())[ii,0])
                counter+=1
