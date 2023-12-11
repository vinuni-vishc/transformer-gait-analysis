import torch
import torch.nn as nn
import os

from model.Transformer import SpatioTemporalTransformer
from GaitDataset import GaitDataset

def weighted_loss(y_pred, y_true, weight, c_i_factor):
    loss_fn = nn.MSELoss(reduction = 'none')
    loss = (loss_fn(y_pred, y_true) * weight).mean()/c_i_factor
    return loss

class TransformerModel:
    def __init__ (self, args):
        self.args = args
        self.doublesided = False
        if self.args.sides == 2:
            self.doublesided = True
        print(f'doublesided = {self.doublesided}')
        print('Loading Data...')
        self.dataset = GaitDataset(target_metric = self.args.target_metric, doublesided = self.doublesided, hef = False)
        self.device = self.dataset.device
        self.model = SpatioTemporalTransformer(num_joints = self.dataset.train_dataset[0].shape[-2], in_chans = 2, joint_embed_dim = 12, num_heads = 2, drop_rate = 0.0).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = weighted_loss
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0 = 40, T_mult = 1, eta_min = self.args.eta_min, verbose = True)

    def train(self):
        X_train, y_train = self.dataset.train_dataset
        X_validation, y_validation = self.dataset.validation_dataset

        print("===================TRAINING===================")

        for epoch in range(self.args.epoch_num):
            for i in range(0, X_train.shape[0], self.args.batch_size):
                self.model.train()
                # Get a batch of data
                x_batch = X_train[i:i + self.args.batch_size]

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                y_pred = self.model(x_batch)
                # Compute the loss 
                y_true = torch.tensor(y_train[i: i + self.args.batch_size], dtype=torch.float32, device = self.device).view(-1)
                loss = weighted_loss(y_pred, y_true, self.dataset.train_weights, self.dataset.c_i_factor)

                # Backward pass and optimization step
                loss.backward()
                self.optimizer.step()
            
            self.scheduler.step()
        
            total_val_loss = 0
            count = 0
            for j in range(0, X_validation.shape[0], self.args.batch_size):
                with torch.no_grad():
                    self.model.eval()
                    x_val_batch = X_validation[j: j + self.args.batch_size]
                    preds = self.model(x_val_batch)
                    batch_loss = weighted_loss(preds, torch.tensor(y_validation[j: j + self.args.batch_size], dtype=torch.float32, device = self.device), self.dataset.validation_weights, self.dataset.c_i_factor)
                    total_val_loss += batch_loss.item() * x_val_batch.size()[0]
                    count += x_val_batch.size()[0]
            val_loss = total_val_loss / count

            self.save(epoch = epoch)

            print(f"Epoch {epoch+1} loss: {loss.item():.4f} validation loss: {val_loss:.4f}")

    def save(self, epoch):
        if not os.path.exists(f"./checkpoints/{self.args.target_metric}"):
            os.makedirs(f"./checkpoints/{self.args.target_metric}")
        torch.save({'epoch': epoch+1,'model_state_dict': self.model.state_dict(),'optimizer_state_dict': self.optimizer.state_dict(), 'loss': self.criterion,}, f"./checkpoints/{self.args.target_metric}/epoch_{epoch+1}.pth")