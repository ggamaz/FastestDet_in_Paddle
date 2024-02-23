import paddle
import paddle.nn as nn

class ShuffleV2Block(nn.Layer):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2D(inp, mid_channels, 1, 1, 0, bias_attr=False),
            nn.BatchNorm2D(mid_channels),
            nn.ReLU(),
            # dw
            nn.Conv2D(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias_attr=False),
            nn.BatchNorm2D(mid_channels),
            # pw-linear
            nn.Conv2D(mid_channels, outputs, 1, 1, 0, bias_attr=False),
            nn.BatchNorm2D(outputs),
            nn.ReLU(),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2D(inp, inp, ksize, stride, pad, groups=inp, bias_attr=False),
                nn.BatchNorm2D(inp),
                # pw-linear
                nn.Conv2D(inp, inp, 1, 1, 0, bias_attr=False),
                nn.BatchNorm2D(inp),
                nn.ReLU(),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = self.channel_shuffle(old_x)
            return paddle.concat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return paddle.concat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.shape
        assert (num_channels % 4 == 0)
        x = x.reshape([batchsize * num_channels // 2, 2, height * width])
        x = x.transpose((1, 0, 2))
        x = x.reshape([2, -1, num_channels // 2, height, width])
        return x[0], x[1]

class ShuffleNetV2(nn.Layer):
    def __init__(self, stage_repeats, stage_out_channels, load_param):
        super(ShuffleNetV2, self).__init__()

        self.stage_repeats = stage_repeats
        self.stage_out_channels = stage_out_channels

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2D(3, input_channel, 3, 2, 1, bias_attr=False),
            nn.BatchNorm2D(input_channel),
            nn.ReLU(),
        )

        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        stage_names = ["stage2", "stage3", "stage4"]
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]
            stageSeq = []
            for i in range(numrepeat):
                if i == 0:
                    stageSeq.append(ShuffleV2Block(input_channel, output_channel, 
                                                mid_channels=output_channel // 2, ksize=3, stride=2))
                else:
                    stageSeq.append(ShuffleV2Block(input_channel // 2, output_channel, 
                                                mid_channels=output_channel // 2, ksize=3, stride=1))
                input_channel = output_channel
            setattr(self, stage_names[idxstage], nn.Sequential(*stageSeq))
        
        if load_param == False:
            self._initialize_weights()
        else:
            print("load param...")

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        P1 = self.stage2(x)
        P2 = self.stage3(P1)
        P3 = self.stage4(P2)

        return P1, P2, P3

    def _initialize_weights(self):
        paddle.disable_static()
        print("Initialize params from: ./module/shufflenetv2_paddle.pdparams")
        self.set_state_dict(paddle.load("./module/shufflenetv2_paddle.pdparams"))

if __name__ == "__main__":
    model = ShuffleNetV2([4, 8, 4], [-1, 24, 48, 96, 192], False)
    load_param = paddle.load("./module/shufflenetv2_paddle.pdparams")
    
    # print(dir(load_param))
    # print(load_param.keys())
    # model.set_state_dict(load_param)
    # model.parameters()
    # model = paddle.vision.models.shufflenet_v2_x0_5(pretrained=True)
    # paddle.summary(model, input_size=(2,3, 352, 352))