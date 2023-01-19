from instSeg.nets import unet, resnet

supported_nets = ['uNet', 'ResNet50', 'ResNet101', 'ResNet152', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7']
supported_nets = [net.lower() for net in supported_nets]

def net_factory(x, config):

    net = config.backbone.lower()
    
    assert net in supported_nets

    if net == 'unet':
        backbone, y = unet(x, filters=config.filters,
                           stages=config.nstage,
                           convs=config.stage_conv,
                           drop_rate=config.dropout_rate,
                           normalization=config.net_normalization,
                           up_scaling=config.up_scaling,
                           concatenate=config.concatenate)
    if net.lower().startswith('resnet'):
        backbone, y = resnet(x,
                            filters=config.nstage,
                            convs=config.stage_conv,
                            drop_rate=config.dropout_rate,
                            normalization=config.net_normalization,
                            up_scaling=config.up_scaling,
                            version=net,
                            transfer_training=config.transfer_training)

    
    return backbone, y