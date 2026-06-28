from hls4ml.converters.onnx_to_hls import get_onnx_attribute, onnx_handler


@onnx_handler('ReduceMax')
def parse_reducemax_layer(node, input_names, input_shapes, graph):
    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'ChannelReduceMax'
    layer['inputs'] = [input_names[0]]
    layer['outputs'] = list(node.output)

    # In ONNX opset 11, axes is an attribute (list of ints), not an input.
    axes = get_onnx_attribute(node, 'axes')  # e.g., [1] for channel axis

    # keepdims=1 keeps the reduced axis as size 1; keepdims=0 drops it.
    # Both produce the same H*W output elements in the HLS kernel.
    keepdims = get_onnx_attribute(node, 'keepdims')
    if keepdims is None:
        keepdims = 1
    layer['keepdims'] = int(keepdims)

    # input_shapes[0] = [batch, C, H, W]  (ONNX channels-first)
    in_shape = input_shapes[0]

    # Axis 1 in ONNX channels-first [N, C, H, W] is the channel dimension.
    # We only support reducing along the channel axis for spatial 2D inputs.
    if axes is None or list(axes) != [1]:
        raise NotImplementedError(
            f'ChannelReduceMax only supports axes=[1] (channel axis), got axes={axes} in node {node.name}'
        )

    layer['n_chan'] = in_shape[1]
    layer['in_height'] = in_shape[2]
    layer['in_width'] = in_shape[3]

    return layer
