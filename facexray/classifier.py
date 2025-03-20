
import segmentation_models_pytorch as smp
import pytorch_lightning as pl

class Classifier(pl.LightningModule):

    def __init__(self):
        super().__init__()
        # Resnet config
        aux_params=dict(
            pooling='max',             # one of 'avg', 'max'
            dropout=0.5,               # dropout ratio, default is None
            activation='sigmoid',      # activation function, default is None
            classes=1,                 # define number of output labels
        )
        self.model = smp.DeepLabV3Plus(encoder_name="efficientnet-b6", encoder_weights="imagenet", in_channels=3, classes=1, aux_params=aux_params)
        
    def forward(self, image):
        output_mask, output_target = self.model(image)
        return output_mask, output_target