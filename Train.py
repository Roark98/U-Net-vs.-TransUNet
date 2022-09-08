import torch
import torch.nn as nn
from torchvision import transforms
import json
from Utils import CustomDataset, dice_score, get_model, dice_score_2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse


def train_model(model_name, model, criterion, optimizer, num_epochs=3):
    tr_losses = {}  #Training losses
    ts_losses = {}  #Test losses
    ts_dice = {}    #Test Dice Scores
    #Epochs
    for epoch in range(num_epochs):
        print('-' * 100)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                epoch_mean_dice = 0

            running_loss = 0.0
            #Batches
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                #Model output
                outputs = model(inputs.float())
                if model_name=='TransUNet':
                    outputs = torch.sigmoid(outputs)
                loss = criterion(outputs, labels.float())

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    out_cut = np.copy(outputs.data.cpu().numpy())
                    #Applying Thresholding
                    out_cut[np.nonzero(out_cut < 0.5)] = 0.0
                    out_cut[np.nonzero(out_cut >= 0.5)] = 1.0
                    #Dice Score Calculation
                    batch_dice = dice_score(
                        out_cut, labels.cpu().numpy()).item()
                    epoch_mean_dice += batch_dice

                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(image_datasets[phase])

            #Storing loss and dice score
            if phase == 'train':
                tr_losses[epoch] = epoch_loss
            else:
                ts_losses[epoch] = epoch_loss
                epoch_mean_dice = epoch_mean_dice/len(dataloaders['test'])

                #Checking if a new best model is found
                if len(ts_dice) == 0 or epoch_mean_dice>max(ts_dice.values()):
                    print('-----New best model found! Saving...')
                    best_model = model

                ts_dice[epoch] = epoch_mean_dice
                print("*****Epoch Test Dice Score: ", epoch_mean_dice)

            print('*****Phase: {} - Loss: {:.4f}'.format(phase, epoch_loss))

    print('-' * 100)
    print("*****Best test loss: "+str(min(ts_losses.values())))

    #Saving releveant records
    if os.path.isdir('./results')==False:
        os.mkdir('./results')
    if os.path.isdir('./results/'+model_name)==False:
        os.mkdir('./results/'+model_name)

    torch.save(best_model.state_dict(), './results/'+model_name+'/trained_model.pth')

    with open('./results/'+model_name+'/train_losses.json', 'w') as f:
        json.dump(tr_losses, f)
    with open('./results/'+model_name+'/test_losses.json', 'w') as f:
        json.dump(ts_losses, f)
    with open('./results/'+model_name+'/Dice_scores.json', 'w') as f:
        json.dump(ts_dice, f)

    #Plotting
    plt.plot(ts_losses.keys(),ts_losses.values())
    plt.plot(tr_losses.keys(),tr_losses.values())
    plt.title('Test Loss Evolution')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str,
                        default='UNet', help='Model to use', choices=['UNet', 'TransUNet'])
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')

    args = parser.parse_args()

    #Data paths
    path_imgs = './Task02_Heart/ImgTr'    #Images
    path_lb = './Task02_Heart/LblTr'    #Labels

    images = os.listdir(path_imgs)
    labels = os.listdir(path_lb)

    #Train-test split
    X_train, X_test, y_train, y_test = train_test_split(images,labels, test_size=0.2, shuffle=True)
    print('Train len: ',len(X_train))
    print('Test len: ',len(X_test))


    #Datasets and Dataloaders
    transformations = transforms.Compose([
        transforms.ToTensor()
    ])
    tr_dataset = CustomDataset(path_imgs, path_lb, X_train, y_train, transformations)
    ts_dataset = CustomDataset(path_imgs, path_lb, X_test, y_test, transformations)

    image_datasets = {'train': tr_dataset, 'test': ts_dataset}

    tr_dataloader = torch.utils.data.DataLoader(
        tr_dataset, batch_size=10, shuffle=True, num_workers=3)
    ts_dataloader = torch.utils.data.DataLoader(
        ts_dataset, batch_size=10, shuffle=True, num_workers=3)

    dataloaders = {'train': tr_dataloader, 'test': ts_dataloader}

    #Model and Hyperparameters
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    device = torch.device('cpu')

    print("Selected device: ", device)

    model = get_model(args.model).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00006)

    #Train and Save
    train_model = train_model(args.model, model, criterion, optimizer, args.epochs)
    
