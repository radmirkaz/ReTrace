import torch

import src.models.effnet


def weights_init_kaiming(m):
    """
    Initialize network weights using Kaiming initialization.
    
    Args:
        m: Module to initialize
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, "bias") and m.bias is not None:
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    """
    Initialize classifier weights with normal distribution.
    
    Args:
        m: Module to initialize
    """
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, std=0.001)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ClassBlock(torch.nn.Module):
    """
    Classification block that takes features and outputs class predictions.
    Used as the final layers in the ReID model.
    
    Args:
        input_dim: Input feature dimension
        class_num: Number of classes for classification
        droprate: Dropout probability
        relu: Whether to use ReLU activation
        bnorm: Whether to use batch normalization
        linear: Size of the linear layer (-1 to use input_dim)
        classify: Whether to include the final classification layer
    """
    def __init__(
        self, input_dim, class_num, droprate, relu=False, bnorm=True, linear=512, classify=True
    ):
        super(ClassBlock, self).__init__()
        self.classify = classify
        add_block = []
        if linear > 0:
            add_block += [torch.nn.Linear(input_dim, linear)]
        else:
            linear = input_dim
        if bnorm:
            add_block += [torch.nn.BatchNorm1d(linear)]
        if relu:
            add_block += [torch.nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [torch.nn.Dropout(p=droprate)]
        add_block = torch.nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [torch.nn.Linear(linear, class_num)]
        classifier = torch.nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        """
        Forward pass through the classification block.
        
        Args:
            x: Input features
            
        Returns:
            Tuple of (class_predictions, features)
        """
        features = self.add_block(x)
        x = self.classifier(features) if self.classify else -1
        return x, features


class EffNetv2(torch.nn.Module):
    """
    EfficientNet V2 model for vehicle re-identification and classification.
    
    Args:
        class_num: Number of vehicle classes (make/model combinations)
        features_dim: Dimension of the output feature vector
        mix_prec: Whether to use mixed precision training
        classify: Whether to include classification head
        imagenet_pretrain: Whether to use ImageNet pretrained weights
        model_name: EfficientNet model variant ('s', 'm', 'l')
    """
    def __init__(
        self,
        class_num=9630,
        features_dim=2048,
        mix_prec=True,
        classify=True,
        imagenet_pretrain=False,
        model_name="s",
    ):
        super().__init__()
        self.mix_prec = mix_prec
        self.backbone = src.models.effnet.EfficientNetV2(
            model_name, n_classes=features_dim, pretrained=imagenet_pretrain
        )
        self.classifier = ClassBlock(1280, class_num, 0.5, linear=features_dim, classify=classify)
        self.EMBEDDING_DIM = features_dim

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input images
            
        Returns:
            Tuple of (class_predictions, features)
        """
        features = self.backbone(x)
        if self.mix_prec:
            features = features.float()

        return self.classifier(features)