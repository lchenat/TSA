#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *

### tsa ###

class MLPBody(nn.Module):
    def __init__(self, input_dim, feature_dim=512, hidden_dim=512):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(input_dim, hidden_dim))
        self.fc2 = layer_init(nn.Linear(hidden_dim, feature_dim))
        self.feature_dim = feature_dim

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x.view(x.size(0), -1))))

class TSAConvBody(nn.Module):
    def __init__(self, in_channels=12, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.conv1_1 = layer_init(nn.Conv2d(in_channels, 32,  kernel_size=3, padding=1)) # 16->16
        self.conv1_2 = layer_init(nn.Conv2d(32, 32, stride=2, kernel_size=3, padding=1)) # 16->8
        self.conv2_1 = layer_init(nn.Conv2d(32, 32,           kernel_size=3, padding=1)) # 8->8
        self.conv2_2 = layer_init(nn.Conv2d(32, 64, stride=2, kernel_size=3, padding=1)) # 8->4
        self.conv3_1 = layer_init(nn.Conv2d(64, 64,           kernel_size=3, padding=1)) # 4->4
        self.conv3_2 = layer_init(nn.Conv2d(64, 128, stride=2,kernel_size=3, padding=1)) # 4->2
        self.conv4_1 = layer_init(nn.Conv2d(128, 128,         kernel_size=3, padding=1)) # 2->2
        self.conv4_2 = layer_init(nn.Conv2d(128, 128,         kernel_size=3, padding=1)) # 2->2
        self.fc = layer_init(nn.Linear(2 * 2 * 128, self.feature_dim))

    def forward(self, x): # you must add relu between every ConvNet!
        y = F.relu(self.conv1_2(F.relu(self.conv1_1(x))))
        y = F.relu(self.conv2_2(F.relu(self.conv2_1(y))))
        y = F.relu(self.conv3_2(F.relu(self.conv3_1(y))))
        y = F.relu(self.conv4_2(F.relu(self.conv4_1(y))))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc(y))
        return y

class LargeTSAMiniConvBody(nn.Module):
    def __init__(self, in_channels=12, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, stride=2, kernel_size=3, padding=1)) # 32->16
        self.conv2 = layer_init(nn.Conv2d(32, 64, stride=2, kernel_size=3, padding=1)) # 16->8
        self.conv3 = layer_init(nn.Conv2d(64, 128, stride=2,kernel_size=3, padding=1)) # 8->4
        self.conv4 = layer_init(nn.Conv2d(128, 128, stride=2,kernel_size=3, padding=1)) # 4->2
        self.fc = layer_init(nn.Linear(2 * 2 * 128, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = F.relu(self.conv4(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc(y))
        return y

class TSAMiniConvBody(nn.Module):
    def __init__(self, in_channels=12, feature_dim=512, scale=1, gate=F.relu): # scale only works for 2^n
        super().__init__()
        self.feature_dim = feature_dim
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, stride=2, kernel_size=3, padding=1)) # 16->8
        self.conv2 = layer_init(nn.Conv2d(32, 64, stride=2, kernel_size=3, padding=1)) # 8->4
        self.conv3 = layer_init(nn.Conv2d(64, 128, stride=2,kernel_size=3, padding=1)) # 4->2
        self.fc = layer_init(nn.Linear(2 * scale * 2 * scale * 128, self.feature_dim))
        self.gate = gate

    def forward(self, x):
        y = self.gate(self.conv1(x))
        y = self.gate(self.conv2(y))
        y = self.gate(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = self.gate(self.fc(y))
        return y

class TSAMiniMiniConvBody(nn.Module):
    def __init__(self, in_channels=12, feature_dim=512, scale=1, gate=F.relu): # scale only works for 2^n
        super().__init__()
        self.feature_dim = feature_dim
        self.scale = scale
        if scale == 1:
            self.conv1 = layer_init(nn.Conv2d(in_channels, 32, stride=2, kernel_size=3, padding=0)) # 11 -> 5
            self.conv2 = layer_init(nn.Conv2d(32, 64, stride=2, kernel_size=3, padding=0)) # 5 -> 2
            self.fc = layer_init(nn.Linear(2 * 2 * 64, self.feature_dim))
        elif scale == 2:
            self.conv1 = layer_init(nn.Conv2d(in_channels, 32, stride=2, kernel_size=4, padding=0)) # 22 -> 10
            self.conv2 = layer_init(nn.Conv2d(32, 64, stride=2, kernel_size=4, padding=0)) # 10 -> 4
            self.conv3 = layer_init(nn.Conv2d(64, 128, stride=2,kernel_size=3, padding=1)) # 4->2
            self.fc = layer_init(nn.Linear(2 * 2 * 128, self.feature_dim))
        else:
            raise Exception('unsupported scale')
        self.gate = gate

    def forward(self, x):
        y = self.gate(self.conv1(x))
        y = self.gate(self.conv2(y))
        if self.scale == 2:
            y = self.gate(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = self.gate(self.fc(y))
        return y

class TSAMiniConvFCBody(nn.Module):
    def __init__(self, in_channels=12, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, stride=2, kernel_size=3, padding=1)) # 16->8
        self.conv2 = layer_init(nn.Conv2d(32, 64, stride=2, kernel_size=3, padding=1)) # 8->4
        self.conv3 = layer_init(nn.Conv2d(64, 128, stride=2,kernel_size=3, padding=1)) # 4->2
        self.fc = layer_init(nn.Linear(2 * 2 * 128, 512))
        self.fc2 = layer_init(nn.Linear(2 * 2 * 128, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc(y))
        y = F.relu(self.fc2(y))
        return y

class TSAOneConvBody(nn.Module):
    def __init__(self, in_channels=12, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, stride=1, kernel_size=3)) # 16->14
        self.conv2 = layer_init(nn.Conv2d(32, 32, stride=1, kernel_size=3)) # 14->12
        self.conv3 = layer_init(nn.Conv2d(32, 64, stride=2, kernel_size=4)) # 12->5
        self.conv4 = layer_init(nn.Conv2d(64, 128, stride=2,kernel_size=3)) # 5->2
        self.fc = layer_init(nn.Linear(2 * 2 * 128, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = F.relu(self.conv4(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc(y))
        return y

class UnetEncoder(nn.Module):
    def __init__(self, in_channels=12, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, stride=2, kernel_size=3, padding=1)) # 16->8
        self.conv2 = layer_init(nn.Conv2d(32, 64, stride=2, kernel_size=3, padding=1)) # 8->4
        self.conv3 = layer_init(nn.Conv2d(64, 128, stride=2,kernel_size=3, padding=1)) # 4->2
        self.fc = layer_init(nn.Linear(2 * 2 * 128, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc(y))
        return y

class UnetDecoder(nn.Module):
    def __init__(self, out_channels=12, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.fc = layer_init(nn.Linear(self.feature_dim, 2 * 2 * 128))

        self.deconv1 = deconv(128, 64, 4) # 2->4
        self.deconv2 = deconv(64, 32, 4) # 4->8
        self.deconv3 = deconv(32, out_channels, 4) # 8->16

    def forward(self, x):
        y = F.relu(self.fc(x)).view(x.size(0), 128, 2, 2)
        y = F.relu(self.deconv1(y))
        y = F.relu(self.deconv2(y))
        y = F.softmax(self.deconv3(y), dim=1)
        return y

### end of tsa ###

class NatureConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

class DDPGConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(DDPGConvBody, self).__init__()
        self.feature_dim = 39 * 39 * 32
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

    def forward(self, x):
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = y.view(y.size(0), -1)
        return y

class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim, ) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x

class TwoLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=(64, 64), gate=F.relu):
        super(TwoLayerFCBodyWithAction, self).__init__()
        hidden_size1, hidden_size2 = hidden_units
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_size1))
        self.fc2 = layer_init(nn.Linear(hidden_size1 + action_dim, hidden_size2))
        self.gate = gate
        self.feature_dim = hidden_size2

    def forward(self, x, action):
        x = self.gate(self.fc1(x))
        phi = self.gate(self.fc2(torch.cat([x, action], dim=1)))
        return phi

class OneLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units, gate=F.relu):
        super(OneLayerFCBodyWithAction, self).__init__()
        self.fc_s = layer_init(nn.Linear(state_dim, hidden_units))
        self.fc_a = layer_init(nn.Linear(action_dim, hidden_units))
        self.gate = gate
        self.feature_dim = hidden_units * 2

    def forward(self, x, action):
        phi = self.gate(torch.cat([self.fc_s(x), self.fc_a(action)], dim=1))
        return phi

class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x

