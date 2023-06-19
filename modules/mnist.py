import torch
import torch.nn as nn

from .gradient_reversal_layer import GradientReversal

__author__ = 'Shayan Gharib'
__docformat__ = 'reStructuredText'
__all__ = ['MNISTFeatureExtractor', 'MNISTClassifier', 'MNISTDiscriminator']


class MNISTFeatureExtractor(nn.Module):
    def __init__(self, input_channels):
        super(MNISTFeatureExtractor, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.shape[0], -1)

        return x


class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=768, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class MNISTDiscriminator(nn.Module):
    def __init__(self, nb_output):
        super(MNISTDiscriminator, self).__init__()
        self.grl = GradientReversal()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=768, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=nb_output)
        )

    def forward(self, x, do_grl=True, alpha=None):
        if do_grl:
            x = self.grl(x, alpha)
        x = self.discriminator(x)
        return x


def main():
    rand_tensor = torch.rand((2, 3, 28, 28))
    model = MNISTFeatureExtractor(input_channels=3)
    cls = MNISTClassifier()

    out = model(rand_tensor)
    print(out.shape)
    out = cls(out)
    print(out.shape)


if __name__ == "__main__":
    main()


# EOF

