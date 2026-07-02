from torch import nn


class SiameseNetwork(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, q1, q2):
        embedded_q1 = self.encoder(q1)
        embedded_q2 = self.encoder(q2)

        return embedded_q1, embedded_q2
