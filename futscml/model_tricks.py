
def conv_swap_channels_inplace(conv_layer, new_order):
    conv_layer.weight.data[:, :, :, :] = conv_layer.weight.data[:, new_order, :, :]
    # Bias is unchanged
    # if conv_layer.bias is not None:
    #     conv_layer.bias.data[:, :, :, :] = conv_layer.weight.data[:, new_order, :, :]
    return conv_layer