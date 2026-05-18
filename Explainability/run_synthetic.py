import numpy as np
import pandas as pd
from utils import *
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
#### DataLoader

def generateData(d, N, seed, ndrift):
    np.random.seed(seed)
    mu1 = np.random.randn(d)
    ind = list(np.random.choice(np.arange(0, d), ndrift, replace=False))
    severity = np.random.normal(2, 1, ndrift)
    print(severity)
    mu2 = mu1.copy()
    mu2[ind] = mu2[ind] + severity

    Sigma = np.eye(d)
    Sigma_y = Sigma.copy()
    #Sigma_y[0, 0] = 1
    X = np.random.multivariate_normal(mu1, Sigma, size=N)
    Y = np.random.multivariate_normal(mu2, Sigma_y, size=N)
    return ind, severity, X, Y

class SyntheticDataset(Dataset):
    def __init__(self, d, N, seed, ndrift):
        # Generate data
        self.ind, self.severity, self.X, self.Y = generateData(d, N, seed, ndrift)
        # Convert data to torch tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.float32)
        
        # Labels: 0 for X samples, 1 for Y samples
        self.labels_X = torch.zeros(len(self.X), dtype=torch.long)
        self.labels_Y = torch.ones(len(self.Y), dtype=torch.long)
        
        # Combine X and Y with their respective labels
        self.data = torch.cat((self.X, self.Y), dim=0)
        self.labels = torch.cat((self.labels_X, self.labels_Y), dim=0)

    def __len__(self):
        # Total number of samples (sum of X and Y samples)
        return len(self.data)

    def __getitem__(self, idx):
        # Return a sample and its label as a tuple
        return self.data[idx], self.labels[idx]
    
############
#Model
############
from torch import nn
import torch.optim as optim


class ClassificationNet(nn.Module):
    def __init__(self, input_dim):
        super(ClassificationNet, self).__init__()
        # self.fc1 = nn.Linear(input_dim, 64)    # First fully connected layer
        # self.fc2 = nn.Linear(64, 32)           # Second fully connected layer
        # self.fc3 = nn.Linear(32, 1)            # Output layer for binary classification
        self.fc1 = nn.Linear(input_dim, 128)  # Increased to 128 units
        self.fc2 = nn.Linear(128, 64)         # Increased to 64 units
        self.fc3 = nn.Linear(64, 32)          # Additional hidden layer
        self.fc4 = nn.Linear(32, 1)  
        self.sigmoid = nn.Sigmoid()            # Sigmoid for binary output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))          # Sigmoid activation for the output
        return x



def train_model(model, dataloader, criterion, optimizer, num_epochs=20):
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data_batch, label_batch in dataloader:
            # Move inputs and labels to device if GPU is used
            label_batch = label_batch.float().unsqueeze(1)  # Reshape for BCELoss

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(data_batch)
            loss = criterion(outputs, label_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate the loss for display
            running_loss += loss.item()

        # Print the average loss for this epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

from captum.attr import IntegratedGradients, GradientShap, KernelShap, DeepLift


def get_dataloader(d,N,seed,ndrift):
    #d = 10    # Number of dimensions
    #N = 5000       # Number of samples per class
    #seed = 44     # Random seed
    #ndrift = 3    # Number of drift dimensions

    # Initialize dataset
    dataset = SyntheticDataset(d, N, seed, ndrift)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size - int(0.1 * len(dataset))
    test_size = int(0.1*len(dataset))

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[train_size, val_size,test_size])

    #len(train_dataset), len(val_dataset), len(test_dataset)

    return train_dataset, val_dataset, test_dataset

from tqdm import tqdm
def training(model,train_dataloader, val_dataloader,loss_fn,optimizer,accuracy): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    accuracy = accuracy.to(device)
    model = model.to(device)
    EPOCHS = 10
    for epoch in tqdm(range(EPOCHS)):
        # Training loop
        train_loss, train_acc = 0.0, 0.0
        for X, y in train_dataloader:
            X, y = X.to(device), y.float().to(device)
            
            model.train()
            
            y_pred = model(X)
            loss = loss_fn(y_pred, y.unsqueeze(1))
            train_loss += loss.item()
            
            acc = accuracy(y_pred, y.unsqueeze(1))
            train_acc += acc
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
            
        # Validation loop
        val_loss, val_acc = 0.0, 0.0
        model.eval()
        with torch.inference_mode():
            for X, y in val_dataloader:
                X, y = X.to(device), y.float().to(device)
            
                y_pred = model(X)
                
                loss = loss_fn(y_pred, y.unsqueeze(1))
                val_loss += loss.item()
                
                acc = accuracy(y_pred, y.unsqueeze(1))
                val_acc += acc
                
            val_loss /= len(val_dataloader)
            val_acc /= len(val_dataloader)
            
        #writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train/loss": train_loss, "val/loss": val_loss}, global_step=epoch)
        #writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train/acc": train_acc, "val/acc": val_acc}, global_step=epoch)
        
        print(f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train acc: {train_acc: .5f}| Val loss: {val_loss: .5f}| Val acc: {val_acc: .5f}")
    return model

def get_explanations(test_dataloader,model):
    return 0

from itertools import product
from torchmetrics.classification import BinaryAccuracy
def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    DIMENSIONS = [10,20]
    NB_DRIFTS = [1,3,7,9]
    N_RUNS = args.n_runs
    
    random_seeds_runs = np.random.RandomState(42).permutation(N_RUNS)

    
    random_seeds_datasets = np.random.RandomState(10).permutation(len(list(product(DIMENSIONS,NB_DRIFTS))))
    
    Scores = {}
    Results = {}
    i = 0
    for d,nb_d in list(product(DIMENSIONS,NB_DRIFTS)):
        print('Start with:',d,nb_d)
        print('Generate Data')
        IG_expl, GS_expl, DL_expl = [],[],[]
        SWD_expl = []
        for seed_run in random_seeds_runs:
            print(seed_run)
            train_dataset, val_dataset, test_dataset = get_dataloader(d,5000,seed_run,nb_d)
            BATCH_SIZE = 32
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,shuffle=True)
            print('Data loaded')
            print('Start training')
            input_dim = d   # Number of features from the dataset
            model_NN1 = ClassificationNet(input_dim).to(device)

            loss_fn = nn.BCELoss().to(device)
            accuracy = BinaryAccuracy().to(device)
            optimizer = optim.Adam(model_NN1.parameters(),lr=0.001)#optim.SGD(model_NN1.parameters(), lr=0.001,momentum=0.9)
            model_NN1 = training(model_NN1,train_dataloader,val_dataloader,loss_fn,optimizer,accuracy) 
            print('Explanations')
            ig = IntegratedGradients(model_NN1)
            gs = GradientShap(model_NN1)
            #ks = KernelShap(model_NN1)
            dl = DeepLift(model_NN1)
            all_attributions_ig = []
            all_labels = []
            all_attributions_gs = []
            all_attributions_ks = []
            all_attributions_dl = []

            for batch_samples, batch_labels in test_dataloader:
                # Ensure samples have the correct shape
                batch_samples = batch_samples.requires_grad_().to(device)  # Enable gradients for attribution
                baseline_dist = torch.zeros((batch_samples.shape[0],input_dim)).to(device)
                #baseline_dist = torch.abs(torch.tensor(Contributions[0],dtype=torch.float32).repeat(batch_samples.shape[0],1))

                # Calculate the attributions for each sample in the batch
                # We use target=0 as we are working with a binary classification output
                attributions, deltas = ig.attribute(batch_samples, target=0, return_convergence_delta=True)

                
                # Append attributions and labels for further analysis
                all_attributions_ig.append(attributions)
                all_labels.append(batch_labels)
                #batch_samples = batch_samples.requires_grad_()
                attributions_gs, deltas = gs.attribute(batch_samples, target=0,  baselines=baseline_dist,return_convergence_delta=True)
                all_attributions_gs.append(attributions_gs)
                
                #batch_samples = batch_samples.requires_grad_()
                attributions_dl = dl.attribute(batch_samples, target=0)
                all_attributions_dl.append(attributions_dl)


            # Concatenate all attributions and labels
            all_attributions_ig = torch.cat(all_attributions_ig, dim=0)  # Shape: [num_samples, num_features]
            all_labels = torch.cat(all_labels, dim=0)

            all_attributions_gs = torch.cat(all_attributions_gs, dim=0)
            all_attributions_dl = torch.cat(all_attributions_dl, dim=0)
            temp = torch.abs(all_attributions_ig[all_labels ==0 ].mean(axis=0)-all_attributions_ig[all_labels ==1 ].mean(axis=0)).detach().cpu().numpy()
            IG_expl.append(temp)
            temp = torch.abs(all_attributions_gs[all_labels ==0 ].mean(axis=0)-all_attributions_gs[all_labels ==1 ].mean(axis=0)).detach().cpu().numpy()
            GS_expl.append(temp)
            temp = torch.abs(all_attributions_dl[all_labels ==0 ].mean(axis=0)-all_attributions_dl[all_labels ==1 ].mean(axis=0)).detach().cpu().numpy()
            DL_expl.append(temp) 

            Syn_samples = []

            all_labels = []
            for batch_samples, batch_labels in test_dataloader:
                Syn_samples.append(batch_samples)
                all_labels.append(batch_labels)
            Syn_samples = torch.cat(Syn_samples, dim=0)
            all_labels = torch.cat(all_labels,dim=0)
            Syn_X = Syn_samples[all_labels==0].numpy()
            Syn_Y = Syn_samples[all_labels==1].numpy()
            rf, betas, SWDs , Contributions = remove_important_features_syn(Syn_X,Syn_Y,3,10000,max_parameter=False,q=0.95)
            #IG_expl.append(all_attributions_ig.detach().numpy())
            #GS_expl.append(all_attributions_gs.detach().numpy())
            #DL_expl.append(all_attributions_dl.detach().numpy())
            SWD_expl.append(np.abs(Contributions[0]))

        Scores[(d,nb_d)] = {'IG': np.vstack(IG_expl), 'GS': np.vstack(GS_expl), 'DL':np.vstack(DL_expl),'SWD':np.vstack(SWD_expl)}
        IG_values, GS_values, DL_values = [],[],[]
        for l in range(len(IG_expl)):
            IG_values.append(cosin_sim(IG_expl[l],SWD_expl[l]))
            GS_values.append(cosin_sim(GS_expl[l],SWD_expl[l]))
            DL_values.append(cosin_sim(DL_expl[l],SWD_expl[l]))
        i+=1

        Results[(d,nb_d)]={'IG':[np.vstack(IG_values).mean(),np.vstack(IG_values).std()],
                           'GS':[np.vstack(GS_values).mean(),np.vstack(GS_values).std()],
                           'DL':[np.vstack(DL_values).mean(),np.vstack(DL_values).std()]}

    print(Results)



    

    ### für jede dim und nb drifts 10 runs

##################
if __name__ == "__main__":
    # Step 1: Create ArgumentParser instance
    parser = argparse.ArgumentParser(description="Synthetic Experiments for drift explanations")

    # Step 2: Define arguments
    parser.add_argument("-n_runs", type=int, help="Number of runs")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("-o", "--output", type=str, default="output.txt", help="Output file name")

    # Step 3: Parse arguments
    args = parser.parse_args()

    # Step 4: Call the main function with parsed arguments
    main(args)

