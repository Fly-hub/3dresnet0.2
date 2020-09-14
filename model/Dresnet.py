import paddle.fluid as fluid
import numpy as np
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import BatchNorm, Linear,Conv3D,Pool2D



class Conv3DBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 padding=1,
                 groups=1,
                 act=None):
        super(Conv3DBNLayer, self).__init__(name_scope)

        self._conv = Conv3D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            act=None,
            bias_attr=False)

        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)

        return y


def conv1x3x3(in_planes, mid_planes, stride=1,act=None):
    return Conv3D(num_channels=in_planes,
                num_filters = mid_planes,
                     filter_size=(1, 3, 3),
                     stride=(1, stride, stride),
                     padding=(0, 1, 1),
                     bias_attr=None,
                     act=act)


def conv3x1x1(mid_planes, planes, stride=1,act =None):
    return Conv3D(mid_planes,
                     planes,
                     filter_size=(3, 1, 1),
                     stride=(stride, 1, 1),
                     padding=(1, 0, 0),
                     bias_attr=None,
                     act=None)


def conv1x1x1(in_planes, out_planes, stride=1,act = None):
    return Conv3D(in_planes,
                     out_planes,
                     filter_size=1,
                     stride=stride,
                     bias_attr=None,
                     act=None)


class BasicBlock(fluid.dygraph.Layer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        n_3d_parameters1 = in_planes * planes * 3 * 3 * 3
        n_2p1d_parameters1 = in_planes * 3 * 3 + 3 * planes
        mid_planes1 = n_3d_parameters1 // n_2p1d_parameters1
        self.conv1_s = conv1x3x3(in_planes, mid_planes1, stride)
        self.bn1_s = BatchNorm(mid_planes1, act='relu')
        self.conv1_t = conv3x1x1(mid_planes1, planes, stride)
        self.bn1_t = BatchNorm(planes, act='relu')

        n_3d_parameters2 = planes * planes * 3 * 3 * 3
        n_2p1d_parameters2 = planes * 3 * 3 + 3 * planes
        mid_planes2 = n_3d_parameters2 // n_2p1d_parameters2
        self.conv2_s = conv1x3x3(planes, mid_planes2)
        self.bn2_s = BatchNorm(mid_planes2, act='relu')
        self.conv2_t = conv3x1x1(mid_planes2, planes)
        self.bn2_t = BatchNorm(planes, act=None)

        #self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1_s(x)
        out = self.bn1_s(out)
        
        out = self.conv1_t(out)
        out = self.bn1_t(out)
        

        out = self.conv2_s(out)
        out = self.bn2_s(out)
       
        out = self.conv2_t(out)
        out = self.bn2_t(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = fluid.layers.elementwise_add(x=out,y=residual)
    
        layer_helper = LayerHelper(self.full_name(), act='relu')
        return layer_helper.append_activation(out)


class Bottleneck(fluid.dygraph.Layer):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, shortcut=True):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = BatchNorm(planes,act='relu')

        n_3d_parameters = planes * planes * 3 * 3 * 3
        n_2p1d_parameters = planes * 3 * 3 + 3 * planes
        mid_planes = n_3d_parameters // n_2p1d_parameters
        self.conv2_s = conv1x3x3(planes, mid_planes, stride)
        self.bn2_s = BatchNorm(mid_planes,act='relu')
        self.conv2_t = conv3x1x1(mid_planes, planes, stride)
        self.bn2_t = BatchNorm(planes,act='relu')

        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = BatchNorm(planes * self.expansion,act=None)
        
        if not shortcut:
         
            self.downsample =fluid.dygraph.Sequential(conv1x1x1(in_planes,
                             planes * self.expansion, stride), BatchNorm(planes * self.expansion,act=None))
       

        self.shortcut =shortcut
        self.stride = stride
        self._num_channels_out = planes * 4

    def forward(self, x):
        #residual = x

        out = self.conv1(x)
        out = self.bn1(out)
      

        out = self.conv2_s(out)
        out = self.bn2_s(out)
       
        out = self.conv2_t(out)
        out = self.bn2_t(out)
     

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.shortcut:
            residual = x
        else:
            residual = self.downsample(x)


        out = fluid.layers.elementwise_add(x=out,y=residual)
    
        layer_helper = LayerHelper(self.full_name(), act='relu')
        return layer_helper.append_activation(out)


class TSNResNet(fluid.dygraph.Layer):
    def __init__(self, name_scope, layers=50, class_dim=101, 
                    
                    n_input_channels=3,
                    seglen =4,
                    seg_num=8,
                    conv1_t_size=7,
                    conv1_t_stride=1,
                    widen_factor=1.0, 
                    no_max_pool=False,
                    weight_decay=None):
        super(TSNResNet, self).__init__(name_scope)

        self.layers = layers
        self.seg_num = seg_num
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_filters = [64, 128, 256, 512]

        block_inplanes = [int(x * widen_factor) for x in num_filters]
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        n_3d_parameters = 3 * self.in_planes * conv1_t_size * 7 * 7
        n_2p1d_parameters = 3 * 7 * 7 + conv1_t_size * self.in_planes
        mid_planes = n_3d_parameters // n_2p1d_parameters
        n_input_channels = seglen*seg_num
        self.conv1_s_bn = Conv3DBNLayer(
            self.full_name(),
            num_channels=n_input_channels,
            num_filters=mid_planes,
            filter_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            act='relu') 

        self.conv1_t_bn = Conv3DBNLayer(
            self.full_name(),
            num_channels=mid_planes,
            num_filters=self.in_planes,
            filter_size=(conv1_t_size, 1, 1),
            stride=(conv1_t_stride, 1, 1),
            padding=(conv1_t_size // 2, 0, 0),
            act='relu')

        #self.maxpool = fluid.layers.pool3d(pool_size=3, pool_stride=2, pool_padding=1,pool_type='max')
        # self.pool2d_max = Pool2D(
        #     pool_size=3,
        #     pool_stride=2,
        #     pool_padding=1,
        #     pool_type='max')

        self.bottleneck_block_list = []
        
        num_channels = 64
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    Bottleneck(
                       
                        in_planes=num_channels,
                        planes=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut))
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                
                shortcut = True

        # self.pool2d_avg = Pool2D(pool_size=7, pool_type='avg', global_pooling=True)
        
        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.out = Linear(input_dim=num_channels,
                          output_dim=class_dim,
                          act='softmax',
                          param_attr=fluid.param_attr.ParamAttr(
                              initializer=fluid.initializer.Uniform(-stdv, stdv)))

    def forward(self, inputs, label=None):
        #out = fluid.layers.reshape(inputs, [-1, inputs.shape[2], inputs.shape[3], inputs.shape[4]])

        y = self.conv1_s_bn(inputs)
        y = self.conv1_t_bn(y)
        if not self.no_max_pool:
           
            y =fluid.layers.pool3d(y,pool_size=3,pool_stride=2,pool_padding=1,pool_type='max')

        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)

        y = fluid.layers.pool3d(y,pool_size=7,pool_type='avg', global_pooling=True)
        
        out = fluid.layers.reshape(x=y, shape=[y.shape[0], -1])

        #out = fluid.layers.reduce_mean(out, dim=1)
        y = self.out(out)

        if label is not None:
            acc = fluid.layers.accuracy(input=y, label=label)
            return y, acc
        else:
            return y


if __name__ == '__main__':
    with fluid.dygraph.guard():
        network = TSNResNet('resnet', 50)
        img = np.zeros([10,3, 3, 224, 224]).astype('float32')
        img = fluid.dygraph.to_variable(img)
        outs = network(img).numpy()
        print(outs.shape)
        #print(network)
