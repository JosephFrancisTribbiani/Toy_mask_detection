import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict


class RunModel:
  def __init__(self, model,
               root: str = './',
               epochs: int = 200, 
               device = 'cuda',
               writer = None, 
               lr: float = 3e-3,
               weight_decay: float = 1e-5,
               step_size: int = 2,
               gamma: float = 0.5):
    self.root = Path(root)
    self.device = device
    self.model = model.to(self.device)
    self.epochs = epochs
    self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
    self.loss_fn = nn.BCELoss(reduction='none')
    self.writer = writer
    self.reset_data()

  def reset_data(self):
    self.hist = defaultdict(list)
    self.hist['best_val_loss'] = float('inf')

  def load_best_model(self, loc: str = 'best_model.pth'):
    checkpoint = torch.load(loc, map_location=self.device)
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f'Лучшая модель была получена на {checkpoint["epoch"]} эпохе\n'
          f'Loss:\t\t{checkpoint["loss"]}')
    self.hist['best_val_loss'] = checkpoint["loss"]

  def train_loop(self, trainloader: torch.utils.data.dataloader.DataLoader):
    self.model.train()
    hist_tr_iter = defaultdict(list)

    for images, labels in trainloader:
      images = images.to(self.device)
      labels = labels.to(self.device)

      self.optimizer.zero_grad()
      outputs = self.model(images)
      num_objects = torch.sum(labels[:, 4])
      loss = torch.sum((self.loss_fn(outputs[:, 0], labels[:, 0]) +
                        self.loss_fn(outputs[:, 1], labels[:, 1]) + 
                        self.loss_fn(outputs[:, 2], labels[:, 2]) + 
                        self.loss_fn(outputs[:, 3], labels[:, 3])) * labels[:, 4]) / num_objects
      has_object_loss = torch.mean(self.loss_fn(outputs[:, 4], labels[:, 4]))
      loss += has_object_loss
      hist_tr_iter['loss'].append(loss.cpu().item())
      loss.backward()
      self.optimizer.step()
    return {key: np.mean(val) for key, val in hist_tr_iter.items()}

  def eval_loop(self, valloader: torch.utils.data.dataloader.DataLoader):
    self.model.eval()
    hist_val_iter  = defaultdict(list)

    with torch.no_grad():
      for images, labels in valloader:
        images = images.to(self.device)
        labels = labels.to(self.device)

        outputs = self.model(images)
        num_objects = torch.sum(labels[:, 4])
        loss = torch.sum((self.loss_fn(outputs[:, 0], labels[:, 0]) +
                          self.loss_fn(outputs[:, 1], labels[:, 1]) + 
                          self.loss_fn(outputs[:, 2], labels[:, 2]) + 
                          self.loss_fn(outputs[:, 3], labels[:, 3])) * labels[:, 4]) / num_objects
        has_object_loss = torch.mean(self.loss_fn(outputs[:, 4], labels[:, 4]))
        loss += has_object_loss
        hist_val_iter['loss'].append(loss.cpu().item())
        return {key: np.mean(val) for key, val in hist_val_iter.items()}

  def fit_model(self,
                trainloader: torch.utils.data.dataloader.DataLoader,
                valloader: torch.utils.data.dataloader.DataLoader,
                reset_data: bool = False, 
                save_model: bool = False,
                save_last: bool = False,
                best_model_name: str = 'best_model.pth', 
                verify_step: int = 5):
    scheduler_state_dict = None
    if reset_data:
      self.reset_data()

    for epoch in tqdm(range(1, self.epochs + 1)):
      hist_tr_iter = self.train_loop(trainloader=trainloader)
      self.hist['train_loss'].append(hist_tr_iter['loss'])

      hist_val_iter = self.eval_loop(valloader=valloader)
      self.hist['val_loss'].append(hist_val_iter['loss'])

      curr_lr = self.optimizer.param_groups[0]["lr"]
      self.hist['lr'].append(curr_lr) 

      if self.scheduler is not None:
        self.scheduler.step()
        scheduler_state_dict = self.scheduler.state_dict()

      if (epoch) % verify_step == 0:
        print(f"Epoch [{epoch}/{self.epochs}]\n"
              f"\tloss TRAIN: {hist_tr_iter['loss']:.4f},\tloss VAL: {hist_val_iter['loss']:.4f}")
        
      if save_model and hist_val_iter['loss'] < self.hist['best_val_loss']:
        torch.save({'epoch': epoch, 
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': hist_val_iter['loss']}, self.root / best_model_name)
        self.hist['best_val_loss'] = hist_val_iter['loss']
      
      if save_last:
        torch.save({'epoch': epoch, 
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': hist_val_iter['loss']}, self.root / 'last_model.pth')

      if self.writer is not None:
        self.writer.add_scalar("Loss/train", hist_tr_iter['loss'], epoch)
        self.writer.add_scalar("Loss/val", hist_val_iter['loss'], epoch)
        self.writer.add_scalar("Learning rate", curr_lr, epoch)

        self.writer.flush()
        self.writer.close()