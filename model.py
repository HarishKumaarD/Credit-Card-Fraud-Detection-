# model.py
import torch
import torch.nn as nn

# --- Original Model ---
class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, 16),
            nn.ReLU(),
            nn.Linear(16, input_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_size):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.fc(x)
        return x

class HybridModel(nn.Module):
    def __init__(self, autoencoder, classifier):
        super(HybridModel, self).__init__()
        self.autoencoder = autoencoder
        self.classifier = classifier

    def forward(self, x):
        x = self.autoencoder.encoder(x)
        x = self.classifier(x)
        return x

# --- NEW: Refined Model V2 with Dropout and an extra layer ---
class AutoencoderV2(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(AutoencoderV2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Dropout(0.2), # ADDED: Dropout layer
            nn.Linear(16, 8),  # ADDED: Extra layer
            nn.ReLU(),
            nn.Linear(8, encoding_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ClassifierV2(nn.Module):
    def __init__(self, input_size):
        super(ClassifierV2, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)

class HybridModelV2(nn.Module):
    def __init__(self, autoencoder, classifier):
        super(HybridModelV2, self).__init__()
        self.autoencoder = autoencoder
        self.classifier = classifier

    def forward(self, x):
        x = self.autoencoder.encoder(x)
        x = self.classifier(x)
        return x