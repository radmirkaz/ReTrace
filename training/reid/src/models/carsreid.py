from torch.utils.data import DataLoader, ConcatDataset
import torch
import pytorch_lightning as pl

import src.models.classifier
import src.utils.losses


class CarsReidentificationTrainEffnet(pl.LightningModule):
    """
    PyTorch Lightning module for training a vehicle re-identification model using EfficientNet.
    
    This class handles the training process including:
    - Model initialization with optional pretrained weights
    - Loss calculation for re-identification and classification
    - Optimization configuration
    
    Attributes:
        lr: Learning rate for optimization
        EMBEDDING_DIM: Dimension of the feature embedding
    """
    lr = 0.00006
    EMBEDDING_DIM = 2048

    def __init__(self, num_classes=9630, pretain_path=None, model_name="m"):
        """
        Initialize the re-identification model.
        
        Args:
            num_classes: Number of vehicle classes for classification
            pretain_path: Path to pretrained model weights (optional)
            model_name: EfficientNet model variant ('s', 'm', 'l')
        """
        super().__init__()

        # Initialize EfficientNet V2 model
        self.net = src.models.classifier.EffNetv2(
            num_classes, self.EMBEDDING_DIM, model_name=model_name
        )

        # Load pretrained weights if provided
        if pretain_path is not None:
            checkpoint = torch.load(pretain_path)
            # Filter out classifier weights to allow different number of classes
            filtered_state_dict = {
                k: v for k, v in checkpoint.items()
                if not k.startswith("classifier.classifier.0")
            }
            self.net.load_state_dict(filtered_state_dict, strict=False)

        # Initialize loss function for re-identification with classification
        self.loss = src.utils.losses.ReIdentificationLossWithClassification()

    def calc_loss(self, out, mask, prefix="Train"):
        """
        Calculate loss and log statistics.
        
        Args:
            out: Model output (logits, features)
            mask: Ground truth labels
            prefix: Prefix for logging ("Train" or "Val")
            
        Returns:
            Loss value
        """
        loss_val, loss_stat = self.loss(out[0], out[1], mask)
        # Prepare loss statistics for logging
        pub_loss_stat = {f"{prefix}/{k}": v for k, v in loss_stat.items()}

        # Configure logging behavior based on training or validation
        on_step = False if prefix == "Val" else True
        on_epoch = not on_step

        # Log statistics to TensorBoard
        self.log_dict(pub_loss_stat, on_step=on_step, on_epoch=on_epoch, add_dataloader_idx=False)
        return loss_val

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of images
            
        Returns:
            Model output (logits, features)
        """
        return self.net(x)

    def training_step(self, batch, batch_nb):
        """
        Single training step.
        
        Args:
            batch: Batch of data (images, labels)
            batch_nb: Batch number
            
        Returns:
            Dictionary containing the loss
        """
        imgs, labels = batch[0], batch[1]
        # Concatenate images and labels from multiple views
        imgs = torch.concat(imgs, axis=0).float()
        labels = torch.concat(labels, axis=0).long()

        # Forward pass and loss calculation
        out = self.forward(imgs)
        return {"loss": self.calc_loss(out, labels)}

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            Configuration dictionary for optimizer and scheduler
        """
        # AdamW optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.05
        )
        # Step learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": True,
                },
            },
        )




# import pytorch_lightning as pl
# import torch
# import numpy as np
# import src.models.classifier
# import src.utils.losses
# from sklearn.preprocessing import normalize


# class CarsReidentification(pl.LightningModule):
#     EMBEDDING_DIM = 2048
#     NUM_CLASSES = 9548

#     def save(self, path):
#         torch.save(self.net, path)
#         torch.save(self.net.backbone, path + ".backbone")

#     def __init__(self, pretain_path=None):
#         super().__init__()

#         if pretain_path is None or str(pretain_path).endswith(".backbone"):
#             self.net = src.models.classifier.MobileVitv2(
#                 self.NUM_CLASSES, self.EMBEDDING_DIM, pretrain_path=pretain_path
#             )
#         else:
#             self.net = torch.load(pretain_path)

#     def forward(self, x):
#         return self.net(x)


# class CarsReidentificationEffnet(pl.LightningModule):
#     EMBEDDING_DIM = 2048
#     NUM_CLASSES = 9548

#     def save(self, path):
#         torch.save(self.net, path)
#         torch.save(self.net.backbone, path + ".backbone")

#     def __init__(self, pretain_path=None, mix_prec=True, classify=True):
#         super().__init__()

#         if pretain_path is None or str(pretain_path).endswith(".backbone"):
#             # self.net = src.models.classifier.EffNetv2(
#             #     class_num=self.NUM_CLASSES, features_dim=self.EMBEDDING_DIM, classify=classify
#             # )
#             self.net = src.models.classifier.EffNetv2S(
#                 class_num=self.NUM_CLASSES,
#                 features_dim=self.EMBEDDING_DIM,
#                 classify=classify,
#                 mix_prec=mix_prec,
#             )
#         else:
#             self.net = torch.load(pretain_path)
#             self.net.classifier.classify = classify

#     def forward(self, x):
#         return self.net(x)


# class CarsReidentificationTrain(CarsReidentification):
#     lr = 0.00006

#     def __init__(self, pretain_path=None):
#         super().__init__(pretain_path)
#         self.loss = src.utils.losses.ReIdentificationLossWithClassification()

#     def calc_loss(self, out, mask, prefix="Train"):
#         loss_val, loss_stat = self.loss(out[0], out[1], mask)
#         pub_loss_stat = {f"{prefix}/{k}": v for k, v in loss_stat.items()}

#         on_step = False if prefix == "Val" else True
#         on_epoch = not on_step

#         self.log_dict(pub_loss_stat, on_step=on_step, on_epoch=on_epoch, add_dataloader_idx=False)
#         return loss_val

#     def forward(self, x):
#         return self.net(x)

#     def training_step(self, batch, batch_nb):
#         imgs, labels = batch[0], batch[1]
#         imgs = torch.concat(imgs, axis=0).float()
#         labels = torch.concat(labels, axis=0).long()

#         out = self.forward(imgs)
#         return {"loss": self.calc_loss(out, labels)}

#     def on_validation_start(self):
#         self.labels = []
#         self.preds = []
#         self.embeddings = []

#     def validation_step(self, batch, batch_idx):
#         imgs, labels = batch[0].float(), batch[1].long()

#         preds, features = self.forward(imgs)

#         for pred, label, embedding in zip(preds, labels, features):
#             self.preds.append(int(pred.argmax().detach().cpu().numpy()))
#             self.labels.append(label)
#             self.embeddings.append(embedding.detach().cpu().numpy())

#     def _find_best_threshold(self, positive_hist, negative_hist):
#         max_acc, max_idx = 0, 0
#         for i in range(1, len(positive_hist) - 1):
#             tp = sum(positive_hist[:i])
#             tn = sum(negative_hist[i:])
#             fp = sum(positive_hist[i:])
#             fn = sum(negative_hist[:i])
#             acc = (tp + tn) / (tp + tn + fp + fn)
#             if acc >= max_acc:
#                 max_acc = acc
#                 max_idx = i

#         return max_idx, max_acc

#     def on_validation_end(self):
#         self.labels = np.array(self.labels, dtype=np.uint32)
#         self.preds = np.array(self.preds, dtype=np.uint32)
#         self.embeddings = normalize(np.array(self.embeddings, dtype=np.float32))

#         step = 0.03
#         max_dist = 3
#         hist_range = int(max_dist / step)

#         same_histogram = np.zeros((hist_range), dtype=np.uint32)
#         other_histogram = np.zeros((hist_range), dtype=np.uint32)

#         for idx in range(len(self.embeddings)):
#             current_embedding = self.embeddings[idx]
#             current_class = self.labels[idx]

#             same_class_mask = self.labels == current_class
#             other_class_mask = np.invert(same_class_mask)
#             same_class_mask[idx] = False

#             same_class_embeddings = self.embeddings[same_class_mask]
#             other_class_embeddings = self.embeddings[other_class_mask]

#             dists_to_same_class = np.linalg.norm(same_class_embeddings - current_embedding, axis=1)
#             dists_to_other_class = np.linalg.norm(
#                 other_class_embeddings - current_embedding, axis=1
#             )

#             dist_to_same_vals = (
#                 (dists_to_same_class * 1 / step).astype(int).clip(0, hist_range - 1)
#             )
#             dist_to_other_vals = (
#                 (dists_to_other_class * 1 / step).astype(int).clip(0, hist_range - 1)
#             )

#             same_histogram[dist_to_same_vals] += 1
#             other_histogram[dist_to_other_vals] += 1

#         best_thresh, acc = self._find_best_threshold(same_histogram, other_histogram)
#         best_thresh = best_thresh * step

#         self.logger.experiment.add_scalar("Val/BestThresh", best_thresh, self.current_epoch)
#         self.logger.experiment.add_scalar("Val/Accuracy", acc, self.current_epoch)

#         self.labels = []
#         self.preds = []
#         self.embeddings = []

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(
#             self.net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.05
#         )
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
#         return (
#             {
#                 "optimizer": optimizer,
#                 "lr_scheduler": {
#                     "scheduler": scheduler,
#                     "interval": "epoch",
#                     "frequency": 1,
#                     "strict": True,
#                 },
#             },
#         )


# class CarsReidentificationTrainEffnet(CarsReidentificationEffnet):
#     lr = 0.00006
#     # lr = 0.000003

#     def __init__(self, pretain_path=None, classify=True):
#         super().__init__(pretain_path, classify)
#         self.loss = src.utils.losses.ReIdentificationLossWithClassification()

#     def calc_loss(self, out, mask, prefix="Train"):
#         loss_val, loss_stat = self.loss(out[0], out[1], mask)
#         pub_loss_stat = {f"{prefix}/{k}": v for k, v in loss_stat.items()}

#         on_step = False if prefix == "Val" else True
#         on_epoch = not on_step

#         self.log_dict(pub_loss_stat, on_step=on_step, on_epoch=on_epoch, add_dataloader_idx=False)
#         return loss_val

#     def forward(self, x):
#         return self.net(x)

#     def training_step(self, batch, batch_nb):
#         imgs, labels = batch[0], batch[1]
#         imgs = torch.concat(imgs, axis=0).float()
#         labels = torch.concat(labels, axis=0).long()

#         out = self.forward(imgs)
#         return {"loss": self.calc_loss(out, labels)}

#     def on_validation_start(self):
#         self.labels = []
#         self.preds = []
#         self.embeddings = []

#     def validation_step(self, batch, batch_idx):
#         imgs, labels = batch[0].float(), batch[1].long()

#         preds, features = self.forward(imgs)

#         for pred, label, embedding in zip(preds, labels, features):
#             self.preds.append(int(pred.argmax().detach().cpu().numpy()))
#             self.labels.append(label)
#             self.embeddings.append(embedding.detach().cpu().float().numpy())

#     def _find_best_threshold(self, positive_hist, negative_hist):
#         max_acc, max_idx = 0, 0
#         for i in range(1, len(positive_hist) - 1):
#             tp = sum(positive_hist[:i])
#             tn = sum(negative_hist[i:])
#             fp = sum(positive_hist[i:])
#             fn = sum(negative_hist[:i])
#             acc = (tp + tn) / (tp + tn + fp + fn)
#             if acc >= max_acc:
#                 max_acc = acc
#                 max_idx = i

#         return max_idx, max_acc

#     def on_validation_end(self):
#         self.labels = np.array([lbl.cpu().item() if isinstance(lbl, torch.Tensor) else lbl for lbl in self.labels], dtype=np.uint32)
#         self.preds = np.array([prd.cpu().item() if isinstance(prd, torch.Tensor) else prd for prd in self.preds], dtype=np.uint32)
#         # self.embeddings = normalize(np.array(self.embeddings, dtype=np.float32))
#         self.embeddings = normalize(np.array(self.embeddings, dtype=np.float32))

#         step = 0.03
#         max_dist = 3
#         hist_range = int(max_dist / step)

#         same_histogram = np.zeros((hist_range), dtype=np.uint32)
#         other_histogram = np.zeros((hist_range), dtype=np.uint32)

#         for idx in range(len(self.embeddings)):
#             current_embedding = self.embeddings[idx]
#             current_class = self.labels[idx]

#             same_class_mask = self.labels == current_class
#             other_class_mask = np.invert(same_class_mask)
#             same_class_mask[idx] = False

#             same_class_embeddings = self.embeddings[same_class_mask]
#             other_class_embeddings = self.embeddings[other_class_mask]

#             dists_to_same_class = np.linalg.norm(same_class_embeddings - current_embedding, axis=1)
#             dists_to_other_class = np.linalg.norm(
#                 other_class_embeddings - current_embedding, axis=1
#             )

#             dist_to_same_vals = (
#                 (dists_to_same_class * 1 / step).astype(int).clip(0, hist_range - 1)
#             )
#             dist_to_other_vals = (
#                 (dists_to_other_class * 1 / step).astype(int).clip(0, hist_range - 1)
#             )

#             same_histogram[dist_to_same_vals] += 1
#             other_histogram[dist_to_other_vals] += 1

#         best_thresh, acc = self._find_best_threshold(same_histogram, other_histogram)
#         best_thresh = best_thresh * step

#         self.logger.experiment.add_scalar("Val/BestThresh", best_thresh, self.current_epoch)
#         self.logger.experiment.add_scalar("Val/Accuracy", acc, self.current_epoch)

#         self.labels = []
#         self.preds = []
#         self.embeddings = []

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(
#             self.net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.05
#         )
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
#         return (
#             {
#                 "optimizer": optimizer,
#                 "lr_scheduler": {
#                     "scheduler": scheduler,
#                     "interval": "epoch",
#                     "frequency": 1,
#                     "strict": True,
#                 },
#             },
#         )
