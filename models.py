import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights
import pytorch_lightning as pl
from torch.optim import Adam
from pytorch_lightning.trainer.supporters import CombinedLoader


class SiameseSigmoid(pl.LightningModule):
    def __init__(self, lr=1e-4, pretrained=True, batchsize=64):
        super().__init__()
        # get hyperparameters learning rate, pretraining and batchsize
        self.lr = lr
        self.pretrained = pretrained
        self.batchsize = batchsize
        self.save_hyperparameters()

        # get pretrained weights for ResNet18 if pretrained=True
        if self.pretrained:
            self.resnet = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet = torchvision.models.resnet18(weights=None)

        # fit the ResNet18 input to the Tiny ImageNet format
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), bias=True)

        # get number of features of fully-connected layer
        self.fc_in_features = self.resnet.fc.in_features

        # prune the last layer of ResNet18 and initialise linear weights
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.resnet.apply(self.init_weights)

        # define fully-connected layer and initialise weights
        self.fc = nn.Sequential(
                nn.Linear(self.fc_in_features * 2, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1),
            )
        self.fc.apply(self.init_weights)

        # define Sigmoid output
        self.sigmoid = nn.Sigmoid()

    def init_weights(self, m):
        # initialise linear weights
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        # siamese part of model
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # perform siamese forward step
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate features and perform rest of forward step
        output = torch.cat((output1, output2), 1)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        images_1, images_2, targets = train_batch
        outputs = self.forward(images_1, images_2).squeeze()
        criterion = nn.BCELoss()
        loss = criterion(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def val_dataloader(self, loader_seen, loader_unseen):
        # combine dataloader of trained and untrained datasets
        loader_seen = loader_seen
        loader_unseen = loader_unseen
        loaders = {"seen": loader_seen, "unseen": loader_unseen}
        combined_loaders = CombinedLoader(loaders, mode="max_size_cycle")
        return combined_loaders

    def validation_step(self, val_batch, batch_idx):
        val_batch_seen = val_batch['seen']
        val_batch_unseen = val_batch['unseen']
        images_1, images_2, targets = val_batch_seen
        images_3, images_4, targets_2 = val_batch_unseen

        # compute validation loss
        outputs = self.forward(images_1, images_2).squeeze()
        criterion = nn.BCELoss()
        loss = criterion(outputs, targets)
        self.log('val_loss', loss)

        # compute accuracy on trained classes
        pred_seen = torch.where(outputs > 0.5, 1, 0)
        correct_seen = torch.sum(pred_seen.eq(targets.view_as(pred_seen)))/len(targets)
        self.log('accuracy_seen', correct_seen)

        # compute accuracy on untrained classes
        outputs_unseen = self.forward(images_3, images_4).squeeze()
        pred_unseen = torch.where(outputs_unseen > 0.5, 1, 0)
        correct_unseen = torch.sum(pred_unseen.eq(targets_2.view_as(pred_unseen)))/len(targets_2)
        self.log('accuracy_unseen', correct_unseen)

        return loss, correct_seen, correct_unseen


class SiameseCosine(pl.LightningModule):
    def __init__(self, lr=1e-4, categories=10, pretrained=True, batchsize=64):
        super().__init__()
        # get hyperparameters learning rate, pretraining and batchsize
        self.lr = lr
        self.pretrained = pretrained
        self.batchsize = batchsize
        self.categories = int(categories)
        self.save_hyperparameters()

        # get pretrained weights for ResNet18 if pretrained=True
        if pretrained:
            self.resnet = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet = torchvision.models.resnet18(weights=None)

        # fit the ResNet18 input to the Tiny ImageNet format
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), bias=True)

        # get number of features of fully-connected layer
        self.fc_in_features = self.resnet.fc.in_features

        # prune the last layer of ResNet18 and initialise linear weights
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.resnet.apply(self.init_weights)

        # define pseudo-classification layer and initialise weights
        self.fc = nn.Sequential(
                nn.Linear(self.fc_in_features, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.categories),
            )
        self.fc.apply(self.init_weights)
        self.softmax = nn.Softmax()

        # define cosine similarity output
        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-8)
        
    def init_weights(self, m):
        # initialise linear weights
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        # siamese part of model
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        output = self.softmax(output)
        return output

    def forward(self, input1, input2):
        # perform siamese forward step
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # compare pseudo-classes with cosine similarity
        output = self.cosine(output1, output2)
        return output
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        images_1, images_2, targets = train_batch
        outputs = self.forward(images_1, images_2).squeeze()
        criterion = nn.BCELoss()
        loss = criterion(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def val_dataloader(self, loader_seen, loader_unseen):
        # combine dataloader of trained and untrained datasets
        loader_seen = loader_seen
        loader_unseen = loader_unseen
        loaders = {"seen": loader_seen, "unseen": loader_unseen}
        combined_loaders = CombinedLoader(loaders, mode="max_size_cycle")
        return combined_loaders

    def validation_step(self, val_batch, batch_idx):
        val_batch_seen = val_batch['seen']
        val_batch_unseen = val_batch['unseen']
        images_1, images_2, targets = val_batch_seen
        images_3, images_4, targets_2 = val_batch_unseen

        # compute validation loss
        outputs = self.forward(images_1, images_2).squeeze()
        criterion = nn.BCELoss()
        loss = criterion(outputs, targets)
        self.log('val_loss', loss)

        # compute accuracy on trained classes
        pred_seen = torch.where(outputs > 0.5, 1, 0)
        correct_seen = torch.sum(pred_seen.eq(targets.view_as(pred_seen)))/len(targets)
        self.log('accuracy_seen', correct_seen)

         # compute accuracy on untrained classes
        outputs_unseen = self.forward(images_3, images_4).squeeze()
        pred_unseen = torch.where(outputs_unseen > 0.5, 1, 0)
        correct_unseen = torch.sum(pred_unseen.eq(targets_2.view_as(pred_unseen)))/len(targets_2)
        self.log('accuracy_unseen', correct_unseen)

        return loss, correct_seen, correct_unseen

    def get_prediction_correctness(self, val_batch, batch_idx):
        # get evaluation of prediction on trained and untrained classes 
        val_batch_seen = val_batch['seen']
        val_batch_unseen = val_batch['unseen']
        images_1, images_2, targets = val_batch_seen
        images_3, images_4, targets_2 = val_batch_unseen

        outputs = self.forward(images_1, images_2).squeeze()
        pred_seen = torch.where(outputs > 0.5, 1, 0)
        prediction_correctness_seen = pred_seen.eq(targets.view_as(pred_seen))

        outputs_unseen = self.forward(images_3, images_4).squeeze()
        pred_unseen = torch.where(outputs_unseen > 0.5, 1, 0)
        prediction_correctness_unseen = pred_unseen.eq(targets_2.view_as(pred_unseen))

        return prediction_correctness_seen, prediction_correctness_unseen