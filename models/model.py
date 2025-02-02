import torch
from torch import nn
from config_loader import ConfigLoader

# parameters
relu_inplace = True
BN_MOMENTUM = 0.1
drop_percent = 0.1

# Load configuration
config = ConfigLoader()
model_config = config.get("model")

def conv_controller(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
    '''
    Every convolution goes through this function
    The point of this is to test other convolution types by replacing only the code here
    '''
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
    return conv

def norm_controller(channels, momentum=BN_MOMENTUM):
    '''
    Every normalization goes through this function
    The point of this is to test other normalization techniques by replacing only the code here
    '''
    # norm = nn.BatchNorm2d(channels, momentum=momentum)
    norm = nn.InstanceNorm2d(channels, momentum=momentum)
    # norm = nn.GroupNorm(num_channels=channels, num_groups=1, affine=False)
    return norm

def act_controller():
    '''
    Every activation goes through this function
    The point of this is to test other activation functions by replacing only the code here
    '''
    act = nn.ReLU(inplace=relu_inplace)
    # act = nn.GELU()
    return act

class BasicBlock(nn.Module):
    '''
    This the basic block of our network. This block is used except for Stem and the first stage
    '''

    def __init__(self, planes):
        super(BasicBlock, self).__init__()

        self.dil1 = nn.Sequential(
            conv_controller(planes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            norm_controller(planes),
            act_controller(),
            
            conv_controller(planes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            norm_controller(planes),
        )

        self.dil2 = nn.Sequential(
            conv_controller(planes, planes, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            norm_controller(planes),
            act_controller(),
            
            conv_controller(planes, planes, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            norm_controller(planes),
        )

        self.dil4 = nn.Sequential(
            conv_controller(planes, planes, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
            norm_controller(planes),
            act_controller(),
            
            conv_controller(planes, planes, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
            norm_controller(planes),
        )

        self.relu = act_controller()

    def forward(self, x):
        residual = x
        dil1 = self.dil1(x)
        dil2 = self.dil2(x)
        dil4 = self.dil4(x)
        out = (dil1 + dil2 + dil4)/3

        out = out + residual
        out = self.relu(out)

        return out

class BottleneckV0(nn.Module):
    '''
    This the bottleneck block. This block is used for the first stage only. This design choise is inspired by ResNet. 
    '''

    expansion = 4
    def __init__(self, inplanes, planes, stride=1):
        super(BottleneckV0, self).__init__()
        self.conv1 = conv_controller(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_controller(planes)

        self.conv2 = conv_controller(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_controller(planes)

        self.conv3 = conv_controller(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_controller(planes * self.expansion)
        self.relu = act_controller()

        self.downsample = nn.Sequential(
                conv_controller(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                norm_controller(planes * self.expansion),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class BottleneckV1(nn.Module):
    '''
    This the bottleneck block. This block is used for the first stage only. 
    The difference between BottleneckV0 and BottleneckV1 is that BottleneckV0 expect an input with 64 channels, while BottleneckV1 expect an input with 256 channels
    '''

    expansion = 4
    def __init__(self, inplanes, planes, stride=1):
        super(BottleneckV1, self).__init__()
        self.conv1 = conv_controller(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_controller(planes)

        self.conv2 = conv_controller(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_controller(planes)

        self.conv3 = conv_controller(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_controller(planes * self.expansion)

        self.relu = act_controller()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + residual
        out = self.relu(out)

        return out

class Stem(nn.Module):
    '''
    This is the stem block. The point of this block is reduce the image size by 1/4 to make the computation later fatser. Also, this blocks learn hierarchy features! 
    '''
    def __init__(self, *args, **kwargs) -> None:
        super(Stem, self).__init__(*args, **kwargs)

        self.conv1 = conv_controller(3, 64, kernel_size=3, stride=2, padding=1,bias=False)
        self.bn1 = norm_controller(64)

        self.conv2 = conv_controller(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = norm_controller(64)

        self.relu = act_controller()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class Trans(nn.Module):
    '''
    This is the transition block. This block do one of three things:
        # down sample the image size and changer the number of kernals 
        # up sample the image size and changer the number of kernals
        # only change the number of kernals
    
    This block is used between stages and substages to perform fusions and transitions
    '''
    def __init__(self, in_planes, planes, mod = "same") -> None:
        super(Trans, self).__init__()
        '''acceptable mod:
        "same", "up", "down" '''
        
        self.mod = mod
        self.conv = None
        if self.mod == "same":
            self.conv = conv_controller(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=1, padding=1)

        elif self.mod == "up":
            self.conv = conv_controller(in_channels=in_planes, out_channels=planes, kernel_size=1, stride=1)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        elif self.mod == "down":
            self.conv = conv_controller(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=2, padding=1)
        else:
            raise Exception("Wrong mod") 
        
        self.bn = norm_controller(planes) 
        self.act = act_controller()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)

        if self.mod == "up":
            out = self.up(out)
        
        return out

class Fuse(nn.Module):
    '''
    This is the fusion block. All streams with different image sizes are combined and exchagne information between stages and substages
    '''
    def __init__(self, n, chs) -> None:
        super(Fuse, self).__init__()
        self.n = n
        self.chs = chs
        self.fusionStreams = nn.ModuleList([])
        
        for i1 in range(self.n):
            fs = nn.ModuleList([])

            for i2 in range((self.n)):

                if i1 == i2:
                    continue

                tss = []
                for i3 in range(abs(i2-i1)):
                    if i2 > i1: #down
                        ts = Trans(self.chs[i1+i3], self.chs[i1+i3+1], "down")
                    else: #up
                        ts = Trans(self.chs[i1-i3], self.chs[i1-i3-1], "up")
                    tss.append(ts)
                
                seq = nn.Sequential(*tss)
                fs.append(seq)

            self.fusionStreams.append(fs)
    
    def forward(self, x):
        out = []
        for i in range(self.n):
            out.append(0) 
        
        for i1 in range(self.n):
            out[i1] = out[i1] + x[i1]
            counter = 0
            for i2 in range(self.n):
                if i2 == i1:
                    continue
                out[i2] = out[i2] + self.fusionStreams[i1][counter](x[i1])
                counter = counter + 1
        return out

class Stream(nn.Module):
    '''
    This is a simple block that create a stram of 4 basic blocks 
    '''
    def __init__(self, channels) -> None:
        super(Stream, self).__init__()

        self.stream = nn.Sequential(
            BasicBlock(channels),
            BasicBlock(channels),
            BasicBlock(channels),
            BasicBlock(channels),
        )

    def forward(self, x):
        return self.stream(x)

class StreamGenerator(nn.Module):
    def __init__(self, n, chs):
        super(StreamGenerator, self).__init__()

        self.n = n
        self.chs = chs

        self.mainStreams = nn.ModuleList([])
        self.transStreams = nn.ModuleList([])

        for i in range(self.n):
            self.mainStreams.append(Stream(self.chs[i]))
            
            if i != self.n-1:
                self.transStreams.append(Trans(self.chs[i], self.chs[i+1], "down"))
        
        self.fusionStreams = Fuse(n=self.n, chs=self.chs)

    def forward(self, x):
        out = []
        for i in range(self.n):
            x = self.mainStreams[i](x)
            out.append(x)

            if i != self.n-1:
                x = self.transStreams[i](x)
            
        out = self.fusionStreams(out)
        return out

class Stage(nn.Module):
    '''
    This is a central block, where a stage is created with all main strams and fusion streams, and all is connected properly 
    '''
    def __init__(self, n, chs) -> None:
        super(Stage, self).__init__()
        self.n = n
        self.chs = chs
        
        self.mainStreams = nn.ModuleList([])
        for i in range(self.n):
            self.mainStreams.append(Stream(self.chs[i]))

        self.fusionStreams = Fuse(n=self.n, chs=self.chs)

    def forward(self, x):
        out = []
        for i in range(self.n):
            out.append(self.mainStreams[i](x[i]))
        out = self.fusionStreams(out)
        return out

class MiniHRNet(nn.Module):
    '''
    This is our model, where everything is combined. 
    '''
    def __init__(self, model_dict, *args, **kwargs) -> None:
        super(MiniHRNet, self).__init__(*args, **kwargs)

        self.num_of_stages = model_dict["num_of_stages"] # * number of streams, ex: 4
        self.stages_rep = model_dict["stages_rep"] # * how many each stage is repeated, ex: 4
        self.num_of_kernals = model_dict["num_of_kernals"] # * number of kernals (channels) for each stage, ex: [48, 96, 144, 192] 
        
        if self.num_of_stages < 2 or self.num_of_stages != len(self.num_of_kernals): #! catching invalid configurations. however, it is not conclusive. so please always check config.
            raise ValueError("Invalid model configuration!")
        
        # Stem and Stage 1
        self.stem = Stem() # * starting the network with the Stem block
        self.stage1 = nn.Sequential( # * creating the bottleneck stage (stage 1)
            BottleneckV0(64, 64),
            BottleneckV0(256, 64),
            BottleneckV0(256, 64),
            BottleneckV0(256, 64),
        )

        self.transition = Trans(256, self.num_of_kernals[0], "same")
        self.streams = StreamGenerator(n=self.num_of_stages, chs=self.num_of_kernals)

        # main stage
        self.stages = nn.ModuleList([]) # * module list for each stage
        for i1 in range(self.stages_rep):
            self.stages.append(Stage(n=self.num_of_stages, chs=self.num_of_kernals)) # * add substages

        
        # Last Layer
        ch_sum = sum(self.num_of_kernals)
        self.last_layer = nn.Sequential( # * segmentation head!
            conv_controller(in_channels=ch_sum, out_channels=ch_sum, kernel_size=1, stride=1, padding=0),
            norm_controller(ch_sum),
            act_controller(),
            nn.Conv2d(in_channels=ch_sum, out_channels=19, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=4, mode='bilinear'),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _, _, ori_height, ori_width = x.size()

        # Steam and Stage 1
        out = self.stem(x)
        out = self.stage1(out)

        out = self.transition(out)
        out = self.streams(out)

        # main stage
        for i1 in range(self.stages_rep):
            out = self.stages[i1](out)

        # Last Layer
        for i in range(self.num_of_stages):
            out[i] = nn.functional.interpolate(input=out[i], size=(int(ori_height/4), int(ori_width/4)),mode='bilinear', align_corners=False)

        out = torch.cat([*out], 1)
        out = self.last_layer(out)

        return out

def get_model():
    """Initialize Mini-HRNet and load weights if available."""
    model = MiniHRNet(model_config)
    if checkpoint := model_config.get("checkpoint"):
        print(f"📂 Loading checkpoint: {checkpoint}")
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model
