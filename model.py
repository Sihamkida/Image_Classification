from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1):
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=output_dim,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=output_dim, out_channels=output_dim,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
        )
        self.relu = nn.ReLU()
        if self.input_dim != self.output_dim:
            self.residual_layer = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim,
                          kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(output_dim),
            )

    def forward(self, input):
        x = self.layer_1(input)
        x = self.relu(x)
        x = self.layer_2(x)
        if self.input_dim != self.output_dim:
            residual = self.residual_layer(input)
        else:
            residual = input
        x += residual
        output = self.relu(x)
        return output


class ConvNet(nn.Module):
    def __init__(self, input_dim=3, input_size=256, output_dim=7, channel_sizes=[16, 32, 64, 128, 256], kernel_size=3, stride=1):
        super(ConvNet, self).__init__()

        self.input_size = input_size
        self.channel_sizes = channel_sizes
        self.num_channels = len(channel_sizes) - 1

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_dim,
                      out_channels=self.channel_sizes[0], kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) / 2)),
            nn.BatchNorm2d(channel_sizes[0]),
            nn.ReLU(),
        )
        self.convolutions = create_cascade_block(
            input_dims=self.channel_sizes[:-1], output_dims=self.channel_sizes[1:], block=BasicBlock, kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) / 2))
        self.pooling = nn.MaxPool2d(kernel_size=2)

        self.output_layer =  nn.Sequential(
            nn.Linear(in_features=int((self.input_size / (2 ** self.num_channels)) * (self.input_size / (2 ** self.num_channels)) * self.channel_sizes[-1]), out_features=self.channel_sizes[-1]),
            nn.BatchNorm1d(channel_sizes[-1]),
            nn.ReLU(),
            nn.Linear(in_features=self.channel_sizes[-1], out_features=output_dim),
        )
    def forward(self, input):

        x = self.input_layer(input)
        
        for block in self.convolutions:
            x = self.pooling(x)
            x = block(x)
            
        x = x.view(-1, int((self.input_size / (2 ** self.num_channels)) *
                   (self.input_size / (2 ** self.num_channels)) * self.channel_sizes[-1]))
        output = self.output_layer(x)
        return output


class ResNet18(nn.Module):
    def __init__(self, output_dim):
        super(ResNet18, self).__init__()
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        self.output_layer =  nn.Sequential(
            nn.Linear(in_features=1000, out_features=output_dim),
        )

    def forward(self, input):
        x = self.model(input)
        output = self.output_layer(x)
        return output



def create_cascade_block(input_dims, output_dims, block, **block_args):
    num_channels = len(input_dims)
    cascade = []
    for n in range(num_channels):
        cascade.append(
            block(input_dim=input_dims[n], output_dim=output_dims[n], **block_args))
    return nn.Sequential(*cascade)
