import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=True, init_type='normal', init_gain=0.02,
             gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf)
    elif netG == 'unet_512':
        net = UnetGenerator(input_nc, output_nc, 9, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################

class GradientDifferenceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, target_real_label=1.0, target_fake_label=0.0):
        super(GradientDifferenceLoss, self).__init__()

    def get_target_tensor(self, prediction, target_is_real):

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, inputs, targets):

        gradient_diff = (inputs.diff(axis=0) - targets.diff(axis=0)).pow(2) + (
                    inputs.diff(axis=1) - targets.diff(axis=1)).pow(2)
        loss_gdl = gradient_diff.sum() / inputs.numel()

        return loss_gdl


class MSE_and_GDL(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MSE_and_GDL, self).__init__()

    def forward(self, inputs, targets, lambda_mse=0, lambda_gdl=0.1):
        squared_error = (inputs - targets).pow(2)
        gradient_diff_i = (inputs.diff(axis=-1) - targets.diff(axis=-1)).pow(2)
        gradient_diff_j = (inputs.diff(axis=-2) - targets.diff(axis=-2)).pow(2)
        loss = (
                           lambda_mse * squared_error.sum() + lambda_gdl * gradient_diff_i.sum() + lambda_gdl * gradient_diff_j.sum()) / inputs.numel()

        return loss


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# good 5
class UnetGenerator(nn.Module):
    '''
        MultiResUNet

        Arguments:
            input_channels {int} -- number of channels in image
            num_classes {int} -- number of segmentation classes
            alpha {float} -- alpha hyperparameter (default: 1.67)

        Returns:
            [keras model] -- MultiResUNet model
            input_nc=3, output_nc=3, num_downs=7, use_dropout=True
        '''

    def __init__(self, input_nc=3, output_nc=3, num_downs=7, use_dropout=True, input_channels=3, num_classes=2,
                 alpha=1.67):
        super().__init__()

        self.alpha = alpha

        # Encoder Path
        self.multiresblock1 = Multiresblock(input_channels, 32)
        self.in_filters1 = int(32 * self.alpha * 0.167) + int(32 * self.alpha * 0.333) + int(32 * self.alpha * 0.5)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.respath1 = Respath(self.in_filters1, 32, respath_length=4)

        self.multiresblock2 = Multiresblock(self.in_filters1, 32 * 2)
        self.in_filters2 = int(32 * 2 * self.alpha * 0.167) + int(32 * 2 * self.alpha * 0.333) + int(
            32 * 2 * self.alpha * 0.5)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.respath2 = Respath(self.in_filters2, 32 * 2, respath_length=3)

        self.multiresblock3 = Multiresblock(self.in_filters2, 32 * 4)
        self.in_filters3 = int(32 * 4 * self.alpha * 0.167) + int(32 * 4 * self.alpha * 0.333) + int(
            32 * 4 * self.alpha * 0.5)
        self.pool3 = torch.nn.MaxPool2d(2)
        self.respath3 = Respath(self.in_filters3, 32 * 4, respath_length=2)

        self.multiresblock4 = Multiresblock(self.in_filters3, 32 * 8)
        self.in_filters4 = int(32 * 8 * self.alpha * 0.167) + int(32 * 8 * self.alpha * 0.333) + int(
            32 * 8 * self.alpha * 0.5)
        self.pool4 = torch.nn.MaxPool2d(2)
        self.respath4 = Respath(self.in_filters4, 32 * 8, respath_length=1)

        self.multiresblock5 = Multiresblock(self.in_filters4, 32 * 16)
        self.in_filters5 = int(32 * 16 * self.alpha * 0.167) + int(32 * 16 * self.alpha * 0.333) + int(
            32 * 16 * self.alpha * 0.5)

        # Decoder path
        self.upsample6 = torch.nn.ConvTranspose2d(self.in_filters5, 32 * 8, kernel_size=(2, 2), stride=(2, 2))
        self.concat_filters1 = 32 * 8 * 2
        self.multiresblock6 = Multiresblock(self.concat_filters1, 32 * 8)
        self.in_filters6 = int(32 * 8 * self.alpha * 0.167) + int(32 * 8 * self.alpha * 0.333) + int(
            32 * 8 * self.alpha * 0.5)

        self.upsample7 = torch.nn.ConvTranspose2d(self.in_filters6, 32 * 4, kernel_size=(2, 2), stride=(2, 2))
        self.concat_filters2 = 32 * 4 * 2
        self.multiresblock7 = Multiresblock(self.concat_filters2, 32 * 4)
        self.in_filters7 = int(32 * 4 * self.alpha * 0.167) + int(32 * 4 * self.alpha * 0.333) + int(
            32 * 4 * self.alpha * 0.5)

        self.upsample8 = torch.nn.ConvTranspose2d(self.in_filters7, 32 * 2, kernel_size=(2, 2), stride=(2, 2))
        self.concat_filters3 = 32 * 2 * 2
        self.multiresblock8 = Multiresblock(self.concat_filters3, 32 * 2)
        self.in_filters8 = int(32 * 2 * self.alpha * 0.167) + int(32 * 2 * self.alpha * 0.333) + int(
            32 * 2 * self.alpha * 0.5)

        self.upsample9 = torch.nn.ConvTranspose2d(self.in_filters8, 32, kernel_size=(2, 2), stride=(2, 2))
        self.concat_filters4 = 32 * 2
        self.multiresblock9 = Multiresblock(self.concat_filters4, 32)
        self.in_filters9 = int(32 * self.alpha * 0.167) + int(32 * self.alpha * 0.333) + int(32 * self.alpha * 0.5)

        self.conv_final = Conv2d_batchnorm(self.in_filters9, num_classes + 1, kernel_size=(1, 1), activation='None')

        ker_s_1 = 16
        pad_1 = int(ker_s_1 / 2 - 1)
        ker_s_2 = 12
        pad_2 = int(ker_s_2 / 2 - 1)
        ker_s_3 = 8
        pad_3 = int(ker_s_3 / 2 - 1)
        ker_s_4 = 20
        pad_4 = int(ker_s_4 / 2 - 1)
        st = 2
        c_channal = 128

        self.wc1 = nn.Parameter(torch.ones([4]))
        self.wc2 = nn.Parameter(torch.ones([4]))
        self.wc3 = nn.Parameter(torch.ones([4]))
        self.wc4 = nn.Parameter(torch.ones([4]))

        self.c_to_p = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Conv2d(input_nc, c_channal, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_channal),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c_channal, output_nc, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_nc)
        )

        # self.c_to_p_big = nn.Sequential(
        #     nn.Conv2d(input_nc, input_nc, 256, 1, 0, bias=False),
        #     nn.BatchNorm2d(3),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(input_nc, output_nc, 256, 1, 0, bias=False),
        #     nn.BatchNorm2d(output_nc)
        # )

        i_c = 3

        self.cov11 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(i_c, 16, ker_s_1, st, pad_1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.cov12 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(i_c, 16, ker_s_2, st, pad_2, bias=False),
            nn.BatchNorm2d(16)
        )
        self.cov13 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(i_c, 16, ker_s_3, st, pad_3, bias=False),
            nn.BatchNorm2d(16)
        )
        self.cov14 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(i_c, 16, ker_s_4, st, pad_4, bias=False),
            nn.BatchNorm2d(16)
        )
        self.se1 = SELayer(64)

        self.cov21 = nn.Sequential(
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, ker_s_1, st, pad_1, bias=False),
            nn.BatchNorm2d(32)
        )
        self.cov22 = nn.Sequential(
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, ker_s_2, st, pad_2, bias=False),
            nn.BatchNorm2d(32)
        )
        self.cov23 = nn.Sequential(
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, ker_s_3, st, pad_3, bias=False),
            nn.BatchNorm2d(32)
        )
        self.cov24 = nn.Sequential(
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, ker_s_4, st, pad_4, bias=False),
            nn.BatchNorm2d(32)
        )
        self.se2 = SELayer(128)

        self.cov31 = nn.Sequential(
            nn.Dropout(0.4),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, ker_s_1, st, pad_1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.cov32 = nn.Sequential(
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, ker_s_2, st, pad_2, bias=False),
            nn.BatchNorm2d(64)
        )
        self.cov33 = nn.Sequential(
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, ker_s_3, st, pad_3, bias=False),
            nn.BatchNorm2d(64)
        )
        self.cov34 = nn.Sequential(
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, ker_s_4, st, pad_4, bias=False),
            nn.BatchNorm2d(64)
        )
        self.se3 = SELayer(256)

        self.cov41 = nn.Sequential(
            nn.Dropout(0.4),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, ker_s_1, st, pad_1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.cov42 = nn.Sequential(
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, ker_s_2, st, pad_2, bias=False),
            nn.BatchNorm2d(128)
        )
        self.cov43 = nn.Sequential(
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, ker_s_3, st, pad_3, bias=False),
            nn.BatchNorm2d(128)
        )
        self.cov44 = nn.Sequential(
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, ker_s_4, st, pad_4, bias=False),
            nn.BatchNorm2d(128)
        )
        self.se4 = SELayer(512)

        ker_s_dc = 4
        pad_dc = int(ker_s_dc / 2 - 1)

        self.cen = nn.Sequential(
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, ker_s_dc, st, pad_dc, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, ker_s_dc, st, pad_dc, bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout(0.2)
        )

        self.dov4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 256, ker_s_dc, st, pad_dc, bias=False),
            nn.BatchNorm2d(256),
            nn.Dropout(0.2)
        )

        self.dov3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 128, ker_s_3, st, pad_3, bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2)
        )
        self.dov2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 64, ker_s_2, st, pad_2, bias=False),
            nn.BatchNorm2d(64),
        )
        self.dov1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, output_nc, ker_s_1, st, pad_1, bias=False),
            # nn.Tanh()
        )

        self.end = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(9, output_nc, 1, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_multires1 = self.multiresblock1(x)
        x_pool1 = self.pool1(x_multires1)
        x_multires1 = self.respath1(x_multires1)

        x_multires2 = self.multiresblock2(x_pool1)
        x_pool2 = self.pool2(x_multires2)
        x_multires2 = self.respath2(x_multires2)

        x_multires3 = self.multiresblock3(x_pool2)
        x_pool3 = self.pool3(x_multires3)
        x_multires3 = self.respath3(x_multires3)

        x_multires4 = self.multiresblock4(x_pool3)
        x_pool4 = self.pool4(x_multires4)
        x_multires4 = self.respath4(x_multires4)

        x_multires5 = self.multiresblock5(x_pool4)

        up6 = torch.cat([self.upsample6(x_multires5), x_multires4], axis=1)
        x_multires6 = self.multiresblock6(up6)

        up7 = torch.cat([self.upsample7(x_multires6), x_multires3], axis=1)
        x_multires7 = self.multiresblock7(up7)

        up8 = torch.cat([self.upsample8(x_multires7), x_multires2], axis=1)
        x_multires8 = self.multiresblock8(up8)

        up9 = torch.cat([self.upsample9(x_multires8), x_multires1], axis=1)
        x_multires9 = self.multiresblock9(up9)

        out = self.conv_final(x_multires9)

        wc11 = torch.exp(self.wc1[0]) / torch.sum(torch.exp(self.wc1))
        wc12 = torch.exp(self.wc1[1]) / torch.sum(torch.exp(self.wc1))
        wc13 = torch.exp(self.wc1[2]) / torch.sum(torch.exp(self.wc1))
        wc14 = torch.exp(self.wc1[3]) / torch.sum(torch.exp(self.wc1))

        wc21 = torch.exp(self.wc2[0]) / torch.sum(torch.exp(self.wc2))
        wc22 = torch.exp(self.wc2[1]) / torch.sum(torch.exp(self.wc2))
        wc23 = torch.exp(self.wc2[2]) / torch.sum(torch.exp(self.wc2))
        wc24 = torch.exp(self.wc2[3]) / torch.sum(torch.exp(self.wc2))

        wc31 = torch.exp(self.wc3[0]) / torch.sum(torch.exp(self.wc3))
        wc32 = torch.exp(self.wc3[1]) / torch.sum(torch.exp(self.wc3))
        wc33 = torch.exp(self.wc3[2]) / torch.sum(torch.exp(self.wc3))
        wc34 = torch.exp(self.wc3[3]) / torch.sum(torch.exp(self.wc3))

        wc41 = torch.exp(self.wc4[0]) / torch.sum(torch.exp(self.wc4))
        wc42 = torch.exp(self.wc4[1]) / torch.sum(torch.exp(self.wc4))
        wc43 = torch.exp(self.wc4[2]) / torch.sum(torch.exp(self.wc4))
        wc44 = torch.exp(self.wc4[3]) / torch.sum(torch.exp(self.wc4))

        # s_wc11 = str(wc11)
        # s_wc12 = str(wc12)
        # s_wc13 = str(wc13)
        # s_wc14 = str(wc14)
        # s_wc21 = str(wc21)
        # s_wc22 = str(wc22)
        # s_wc23 = str(wc23)
        # s_wc24 = str(wc24)
        # s_wc31 = str(wc31)
        # s_wc32 = str(wc32)
        # s_wc33 = str(wc33)
        # s_wc34 = str(wc34)
        # s_wc41 = str(wc41)
        # s_wc42 = str(wc42)
        # s_wc43 = str(wc43)
        # s_wc44 = str(wc44)
        #
        # with open("w_data_Munet_GDL.txt", 'a') as f:
        #     f.write('w1 \n')
        #     f.write("{} {} {} {} \n".format(s_wc11, s_wc12, s_wc13, s_wc14))
        #     f.write('w2 \n')
        #     f.write("{} {} {} {} \n".format(s_wc21, s_wc22, s_wc23, s_wc24))
        #     f.write('w3 \n')
        #     f.write("{} {} {} {} \n".format(s_wc31, s_wc32, s_wc33, s_wc34))
        #     f.write('w4 \n')
        #     f.write("{} {} {} {} \n".format(s_wc41, s_wc42, s_wc43, s_wc44))
        #
        cx1 = self.se1(torch.cat([torch.cat([wc11 * self.cov11(x), wc12 * self.cov12(x)], 1),
                                  torch.cat([wc13 * self.cov13(x), wc14 * self.cov14(x)], 1)], 1))

        # cx1 = self.se1(torch.cat([torch.cat([wc11 * self.cov11(torch.cat([x, out], 1)), wc12 * self.cov12(torch.cat([x, out], 1))], 1),
        #                           torch.cat([wc13 * self.cov13(torch.cat([x, out], 1)), wc14 * self.cov14(torch.cat([x, out], 1))], 1)], 1))

        cx2 = self.se2(torch.cat([torch.cat([wc21 * self.cov21(cx1), wc22 * self.cov22(cx1)], 1),
                                  torch.cat([wc23 * self.cov23(cx1), wc24 * self.cov24(cx1)], 1)], 1))
        cx3 = self.se3(torch.cat([torch.cat([wc31 * self.cov31(cx2), wc32 * self.cov32(cx2)], 1),
                                  torch.cat([wc33 * self.cov33(cx2), wc34 * self.cov34(cx2)], 1)], 1))
        cx4 = self.se4(torch.cat([torch.cat([wc41 * self.cov41(cx3), wc42 * self.cov42(cx3)], 1),
                                  torch.cat([wc43 * self.cov43(cx3), wc44 * self.cov44(cx3)], 1)], 1))

        ce = self.cen(cx4)

        d4 = torch.cat([ce, cx4], 1)
        dx4 = self.dov4(d4)

        d3 = torch.cat([dx4, cx3], 1)
        dx3 = self.dov3(d3)

        d2 = torch.cat([dx3, cx2], 1)
        dx2 = self.dov2(d2)

        d1 = torch.cat([dx2, cx1], 1)
        dx1 = self.dov1(d1)

        #  return dx1

        return self.end(torch.cat([dx1, torch.cat([out, self.c_to_p(x)], 1)], 1))

        # return out


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=True):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        # downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=(31, 31), stride=(2, 2), padding=15, bias=use_bias)
        ker_s_1 = 8
        pad_1 = int(ker_s_1 / 2 - 1)
        st = 2
        # pad_1 = int((ker_s_1 - 1) / 2)

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=(ker_s_1, ker_s_1), stride=(st, st), padding=pad_1,
                             bias=use_bias)
        # downconv = {
        #     nn.Conv2d(input_nc, inner_nc, kernel_size=(1, 1), stride=(0, 0), padding=1, bias=use_bias),
        #     nn.MaxPool2d(kernel_size=2)
        # }

        # downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=(13, 13), stride=(2, 2), padding=6, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        ##############################################
        ker_s_2 = ker_s_1
        pad_2 = pad_1

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=(ker_s_2, ker_s_2), stride=(st, st),
                                        padding=(pad_2, pad_2))
            # upconv = [
            #     nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=(1, 1), stride=(0, 0), padding=(0, 0)),
            #     nn.MaxPool2d(2)
            # ]
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=(ker_s_2, ker_s_2), stride=(st, st),
                                        padding=(pad_2, pad_2), bias=use_bias)
            # upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=(ker_s_2, ker_s_2), stride=(st, st),
                                        padding=(pad_2, pad_2), bias=use_bias)
            # upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # ==============================================
        kw = 4  # kw=4
        padw = 1  # kw=1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),

            # nn.AdaptiveAvgPool2d(1),
            # nn.Conv2d(ndf, ndf // 4, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(ndf // 4),
            # nn.PReLU(ndf // 4),
            # nn.Conv2d(ndf // 4, ndf, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(ndf),
            # nn.PReLU(ndf),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    ##################################上面改进自定义判别器


#########  se
class se_block(nn.Module):
    def __init__(self, channels, ratio=16):
        super(se_block, self).__init__()
        # 空间信息进行压缩
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 经过两次全连接层，学习不同通道的重要性
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio, False),
            nn.ReLU(),
            nn.Linear(channels // ratio, channels, False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # 取出batch size和通道数

        # b,c,w,h->b,c,1,1->b,c 压缩与通道信息学习
        avg = self.avgpool(x).view(b, c)

        # b,c->b,c->b,c,1,1 激励操作
        y = self.fc(avg).view(b, c, 1, 1)

        return x * y.expand_as(x)


# class NLayerDiscriminator(nn.Module):
#     """Defines a PatchGAN discriminator"""
#
#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
#         """Construct a PatchGAN discriminator
#
#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             ndf (int)       -- the number of filters in the last conv layer
#             n_layers (int)  -- the number of conv layers in the discriminator
#             norm_layer      -- normalization layer
#         """
#         super(NLayerDiscriminator, self).__init__()
#
#         if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#         ks = 2  # 4
#         pd = 1  # 1
#
#         self.model1 = nn.Sequential(
#             nn.Conv2d(input_nc, ndf, kernel_size=(ks, ks), stride=(2, 2), padding=(pd, pd), bias=use_bias),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         )
#
#         self.se = SELayer(ndf)
#
#         self.model2 = nn.Sequential(
#             nn.Conv2d(ndf, ndf * 2, kernel_size=(ks, ks), stride=(2, 2), padding=(pd, pd), bias=use_bias),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv2d(ndf * 2, ndf * 4, kernel_size=(ks, ks), stride=(2, 2), padding=(pd, pd), bias=use_bias),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv2d(ndf * 4, ndf * 8, kernel_size=(ks, ks), stride=(2, 2), padding=(pd, pd), bias=use_bias),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Conv2d(ndf * 8, 1, kernel_size=(ks, ks), stride=(2, 2), padding=(pd, pd), bias=use_bias),
#         )
#
#     def forward(self, input):
#         """Standard forward."""
#         input1 = self.model1(input)
#         ins = self.se(input1)
#         add = self.model2(input1) + self.model2(ins)
#         # return add
#         return add


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net1 = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
        )

        self.se = SELayer(ndf * 2)

        self.net2 = nn.Sequential(
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.Sigmoid()
        )

        # self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        x1 = self.net1(input)
        return self.net2(self.se(x1))


class Conv2d_batchnorm(torch.nn.Module):
    '''
    2D Convolutional layers

    Arguments:
        num_in_filters {int} -- number of input filters
        num_out_filters {int} -- number of output filters
        kernel_size {tuple} -- size of the convolving kernel
        stride {tuple} -- stride of the convolution (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})

    '''

    def __init__(self, num_in_filters, num_out_filters, kernel_size, stride=(1, 1), activation='relu'):
        super().__init__()
        self.activation = activation
        self.conv1 = torch.nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters, kernel_size=kernel_size,
                                     stride=stride, padding='same')
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)

        if self.activation == 'relu':
            return torch.nn.functional.relu(x)
        else:
            return x


class Multiresblock(torch.nn.Module):
    '''
    MultiRes Block

    Arguments:
        num_in_channels {int} -- Number of channels coming into mutlires block
        num_filters {int} -- Number of filters in a corrsponding UNet stage
        alpha {float} -- alpha hyperparameter (default: 1.67)

    '''

    def __init__(self, num_in_channels, num_filters, alpha=1.67):
        super().__init__()
        self.alpha = alpha
        self.W = num_filters * alpha

        filt_cnt_3x3 = int(self.W * 0.167)
        filt_cnt_5x5 = int(self.W * 0.333)
        filt_cnt_7x7 = int(self.W * 0.5)
        num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7

        self.shortcut = Conv2d_batchnorm(num_in_channels, num_out_filters, kernel_size=(1, 1), activation='None')

        self.conv_3x3 = Conv2d_batchnorm(num_in_channels, filt_cnt_3x3, kernel_size=(3, 3), activation='relu')

        self.conv_5x5 = Conv2d_batchnorm(filt_cnt_3x3, filt_cnt_5x5, kernel_size=(3, 3), activation='relu')

        self.conv_7x7 = Conv2d_batchnorm(filt_cnt_5x5, filt_cnt_7x7, kernel_size=(3, 3), activation='relu')

        self.batch_norm1 = torch.nn.BatchNorm2d(num_out_filters)
        self.batch_norm2 = torch.nn.BatchNorm2d(num_out_filters)

    def forward(self, x):
        shrtct = self.shortcut(x)

        a = self.conv_3x3(x)
        b = self.conv_5x5(a)
        c = self.conv_7x7(b)

        x = torch.cat([a, b, c], axis=1)
        x = self.batch_norm1(x)

        x = x + shrtct
        x = self.batch_norm2(x)
        x = torch.nn.functional.relu(x)

        return x


class Respath(torch.nn.Module):
    '''
    ResPath

    Arguments:
        num_in_filters {int} -- Number of filters going in the respath
        num_out_filters {int} -- Number of filters going out the respath
        respath_length {int} -- length of ResPath

    '''

    def __init__(self, num_in_filters, num_out_filters, respath_length):

        super().__init__()

        self.respath_length = respath_length
        self.shortcuts = torch.nn.ModuleList([])
        self.convs = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])

        for i in range(self.respath_length):
            if (i == 0):
                self.shortcuts.append(
                    Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size=(1, 1), activation='None'))
                self.convs.append(
                    Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size=(3, 3), activation='relu'))


            else:
                self.shortcuts.append(
                    Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size=(1, 1), activation='None'))
                self.convs.append(
                    Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size=(3, 3), activation='relu'))

            self.bns.append(torch.nn.BatchNorm2d(num_out_filters))

    def forward(self, x):

        for i in range(self.respath_length):
            shortcut = self.shortcuts[i](x)

            x = self.convs[i](x)
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)

            x = x + shortcut
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)

        return x


# class MultiResUnet(torch.nn.Module):
# '''
# MultiResUNet
#
# Arguments:
#     input_channels {int} -- number of channels in image
#     num_classes {int} -- number of segmentation classes
#     alpha {float} -- alpha hyperparameter (default: 1.67)
#
# Returns:
#     [keras model] -- MultiResUNet model
# '''
#
# def __init__(self, input_channels, num_classes, alpha=1.67):
#     super().__init__()
#
#     self.alpha = alpha
#
#     # Encoder Path
#     self.multiresblock1 = Multiresblock(input_channels, 32)
#     self.in_filters1 = int(32 * self.alpha * 0.167) + int(32 * self.alpha * 0.333) + int(32 * self.alpha * 0.5)
#     self.pool1 = torch.nn.MaxPool2d(2)
#     self.respath1 = Respath(self.in_filters1, 32, respath_length=4)
#
#     self.multiresblock2 = Multiresblock(self.in_filters1, 32 * 2)
#     self.in_filters2 = int(32 * 2 * self.alpha * 0.167) + int(32 * 2 * self.alpha * 0.333) + int(
#         32 * 2 * self.alpha * 0.5)
#     self.pool2 = torch.nn.MaxPool2d(2)
#     self.respath2 = Respath(self.in_filters2, 32 * 2, respath_length=3)
#
#     self.multiresblock3 = Multiresblock(self.in_filters2, 32 * 4)
#     self.in_filters3 = int(32 * 4 * self.alpha * 0.167) + int(32 * 4 * self.alpha * 0.333) + int(
#         32 * 4 * self.alpha * 0.5)
#     self.pool3 = torch.nn.MaxPool2d(2)
#     self.respath3 = Respath(self.in_filters3, 32 * 4, respath_length=2)
#
#     self.multiresblock4 = Multiresblock(self.in_filters3, 32 * 8)
#     self.in_filters4 = int(32 * 8 * self.alpha * 0.167) + int(32 * 8 * self.alpha * 0.333) + int(
#         32 * 8 * self.alpha * 0.5)
#     self.pool4 = torch.nn.MaxPool2d(2)
#     self.respath4 = Respath(self.in_filters4, 32 * 8, respath_length=1)
#
#     self.multiresblock5 = Multiresblock(self.in_filters4, 32 * 16)
#     self.in_filters5 = int(32 * 16 * self.alpha * 0.167) + int(32 * 16 * self.alpha * 0.333) + int(
#         32 * 16 * self.alpha * 0.5)
#
#     # Decoder path
#     self.upsample6 = torch.nn.ConvTranspose2d(self.in_filters5, 32 * 8, kernel_size=(2, 2), stride=(2, 2))
#     self.concat_filters1 = 32 * 8 * 2
#     self.multiresblock6 = Multiresblock(self.concat_filters1, 32 * 8)
#     self.in_filters6 = int(32 * 8 * self.alpha * 0.167) + int(32 * 8 * self.alpha * 0.333) + int(
#         32 * 8 * self.alpha * 0.5)
#
#     self.upsample7 = torch.nn.ConvTranspose2d(self.in_filters6, 32 * 4, kernel_size=(2, 2), stride=(2, 2))
#     self.concat_filters2 = 32 * 4 * 2
#     self.multiresblock7 = Multiresblock(self.concat_filters2, 32 * 4)
#     self.in_filters7 = int(32 * 4 * self.alpha * 0.167) + int(32 * 4 * self.alpha * 0.333) + int(
#         32 * 4 * self.alpha * 0.5)
#
#     self.upsample8 = torch.nn.ConvTranspose2d(self.in_filters7, 32 * 2, kernel_size=(2, 2), stride=(2, 2))
#     self.concat_filters3 = 32 * 2 * 2
#     self.multiresblock8 = Multiresblock(self.concat_filters3, 32 * 2)
#     self.in_filters8 = int(32 * 2 * self.alpha * 0.167) + int(32 * 2 * self.alpha * 0.333) + int(
#         32 * 2 * self.alpha * 0.5)
#
#     self.upsample9 = torch.nn.ConvTranspose2d(self.in_filters8, 32, kernel_size=(2, 2), stride=(2, 2))
#     self.concat_filters4 = 32 * 2
#     self.multiresblock9 = Multiresblock(self.concat_filters4, 32)
#     self.in_filters9 = int(32 * self.alpha * 0.167) + int(32 * self.alpha * 0.333) + int(32 * self.alpha * 0.5)
#
#     self.conv_final = Conv2d_batchnorm(self.in_filters9, num_classes + 1, kernel_size=(1, 1), activation='None')
#
# def forward(self, x: torch.Tensor) -> torch.Tensor:
#     x_multires1 = self.multiresblock1(x)
#     x_pool1 = self.pool1(x_multires1)
#     x_multires1 = self.respath1(x_multires1)
#
#     x_multires2 = self.multiresblock2(x_pool1)
#     x_pool2 = self.pool2(x_multires2)
#     x_multires2 = self.respath2(x_multires2)
#
#     x_multires3 = self.multiresblock3(x_pool2)
#     x_pool3 = self.pool3(x_multires3)
#     x_multires3 = self.respath3(x_multires3)
#
#     x_multires4 = self.multiresblock4(x_pool3)
#     x_pool4 = self.pool4(x_multires4)
#     x_multires4 = self.respath4(x_multires4)
#
#     x_multires5 = self.multiresblock5(x_pool4)
#
#     up6 = torch.cat([self.upsample6(x_multires5), x_multires4], axis=1)
#     x_multires6 = self.multiresblock6(up6)
#
#     up7 = torch.cat([self.upsample7(x_multires6), x_multires3], axis=1)
#     x_multires7 = self.multiresblock7(up7)
#
#     up8 = torch.cat([self.upsample8(x_multires7), x_multires2], axis=1)
#     x_multires8 = self.multiresblock8(up8)
#
#     up9 = torch.cat([self.upsample9(x_multires8), x_multires1], axis=1)
#     x_multires9 = self.multiresblock9(up9)
#
#     out = self.conv_final(x_multires9)
#
#     return out

def main():
    img1 = torch.randn(1, 3, 256, 256)
    img2 = torch.randn(64, 3, 256, 256)

    dis_pix_net = PixelDiscriminator(input_nc=3)
    dis_nlay_net = NLayerDiscriminator(input_nc=3)
    gen_unet = UnetGenerator(input_nc=3, output_nc=3, num_downs=7, use_dropout=True)

    print('dis_pix:', dis_pix_net(img1).shape)
    print('dis_nply:', dis_nlay_net(img1).shape)
    # print('gen_unet:', gen_unet(img1).shape)
    #

    # print(dis_nlay_net(img1).shape)
    # print(gen_unet)

    print(gen_unet(img1).shape)
    # print(dis_pix_net)


if __name__ == "__main__":
    main()
