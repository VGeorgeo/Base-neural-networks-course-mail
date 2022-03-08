import torch.nn as nn


def conv_bn_relu(in_planes, out_planes, kernel=3, stride=1, padding=1):
    net = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=padding),
                        nn.BatchNorm2d(num_features=out_planes),
                        nn.ReLU(True))
    return net;


def downsample(in_planes, out_planes):
    net = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=11, stride=2, padding=5),
                        nn.BatchNorm2d(num_features=out_planes),
                        nn.ReLU(True))
    return net;


def upsample(in_planes, out_planes):
    net = nn.Sequential(nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=11, stride=2, padding=5, output_padding=1),
                        nn.BatchNorm2d(num_features=out_planes),
                        nn.ReLU(True))
    return net;


# input size Bx3x224x224
class SegmenterModel(nn.Module):
    def __init__(self, in_size=3):
        super(SegmenterModel, self).__init__()

        self.features = nn.Sequential()

        self.features.add_module('step0', conv_bn_relu(3, 64))
        self.features.add_module('step1', conv_bn_relu(64, 64))
        self.features.add_module('step2', downsample(64, 128))
        self.features.add_module('step3', conv_bn_relu(128, 128))
        self.features.add_module('step4', conv_bn_relu(128, 128))
        self.features.add_module('step5', downsample(128, 256))
        #self.features.add_module('dropout5', nn.Dropout2d(p=0.2))
        self.features.add_module('step6', conv_bn_relu(256, 256))
        #self.features.add_module('dropout6', nn.Dropout2d(p=0.2))
        self.features.add_module('step7', upsample(256, 128))
        #self.features.add_module('dropout7', nn.Dropout2d(p=0.2))
        self.features.add_module('step8', conv_bn_relu(128, 128))
        #self.features.add_module('dropout8', nn.Dropout2d(p=0.2))
        self.features.add_module('step9', conv_bn_relu(128, 128))
        #self.features.add_module('dropout9', nn.Dropout2d(p=0.2))
        self.features.add_module('step10', upsample(128, 64))
        #self.features.add_module('dropout10', nn.Dropout2d(p=0.2))
        self.features.add_module('step11', conv_bn_relu(64, 64))
        #self.features.add_module('dropout11', nn.Dropout2d(p=0.2))
        self.features.add_module('step12', conv_bn_relu(64, 3))
        #self.features.add_module('dropout12', nn.Dropout2d(p=0.2))
        self.features.add_module('step14', nn.Softmax2d())
        #self.features.add_module('step15', nn.ConvTranspose2d(3, 1, kernel_size=3, stride=1, padding=1, output_padding=0))
        #self.features.add_module('step16', nn.BatchNorm2d(num_features=1))
        #self.features.add_module('step17', nn.ReLU(True))


    def forward(self, input):
        #print(input.size())
        input = self.features(input)
        #print(input.size())
        #return input
        return input[:,0,:,:].unsqueeze(1)
        #return torch.sum(input, dim = 1).unsqueeze(1)
