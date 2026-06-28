import numpy as np

from hls4ml.converters.onnx_to_hls import get_constant_value, get_onnx_attribute, onnx_handler


def _get_slice_constant(graph, name):
    """Read a constant value from either a graph initializer or a Constant op output."""
    from onnx import numpy_helper  # local import avoids shadowing by converters/onnx/ subfolder
    # Check initializer first (weights / pre-set constants)
    tensor = next((x for x in graph.initializer if x.name == name), None)
    if tensor is not None:
        return numpy_helper.to_array(tensor)
    # Fall back to Constant op nodes (opset 10+ Slice uses this pattern)
    for node in graph.node:
        if node.op_type == 'Constant' and node.output[0] == name:
            return numpy_helper.to_array(node.attribute[0].t)
    raise RuntimeError(f'Cannot resolve constant tensor: {name}')


@onnx_handler('Transpose')
def parse_transpose_layer(node, input_names, input_shapes, graph):
    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'Transpose'
    layer['inputs'] = input_names
    layer['outputs'] = list(node.output)

    perm = [list(i.ints) for i in node.attribute][0]  # This will get something like [[a,b,c]][0] = [a,b,c]
    layer['perm'] = [x - 1 for x in perm[1:]]  # Ignore the batch dimension in ONNX, and adjust the perm indexing

    return layer


@onnx_handler('Reshape')
def parse_reshape_layer(node, input_names, input_shapes, graph):
    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'Reshape'
    layer['inputs'] = input_names
    layer['outputs'] = list(node.output)

    return layer


@onnx_handler('Flatten')
def parse_flatten_layer(node, input_names, input_shapes, graph):
    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'Reshape'
    layer['inputs'] = input_names
    layer['outputs'] = list(node.output)
    layer['target_shape'] = [-1]  # does not contain batch dimension

    return layer


@onnx_handler('Resize')
def parse_resize_layer(node, input_names, input_shapes, graph):
    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'Resize'
    layer['inputs'] = input_names
    layer['outputs'] = list(node.output)
    layer['in_height'] = input_shapes[0][2]
    layer['in_width'] = input_shapes[0][1]
    layer['out_width'] = input_shapes[0][1]
    layer['out_height'] = input_shapes[0][2]
    layer['n_chan'] = input_shapes[0][3]
    layer['algorithm'] = get_onnx_attribute(node, 'mode')
    # The following is used in initialize() method.
    # Probably a better solution would be to have a channels last parameter at QONNX level
    layer['data_format'] = (
        'channels_last' if any(node.domain == 'qonnx.custom_op.channels_last' for node in graph.node) else 'channels_first'
    )

    return layer


@onnx_handler('Slice')
def parse_slice_layer(node, input_names, input_shapes, graph):
    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'ChannelSlice'
    layer['inputs'] = [input_names[0]]  # only the data tensor; starts/ends/axes are constants
    layer['outputs'] = list(node.output)

    # Read starts, ends, axes from constant op outputs (ONNX opset 10+ format)
    starts = _get_slice_constant(graph, node.input[1])
    ends   = _get_slice_constant(graph, node.input[2])
    axes   = _get_slice_constant(graph, node.input[3]) if len(node.input) > 3 else np.array([1])

    # steps is an optional 5th input (ONNX opset 10+); only step=1 is supported.
    if len(node.input) > 4:
        steps = _get_slice_constant(graph, node.input[4])
        if not all(int(s) == 1 for s in steps):
            raise NotImplementedError(
                f'ChannelSlice only supports step=1, got steps={list(steps)} '
                f'in node {node.name}'
            )

    # input_shapes[0] = [batch, C, H, W]  (ONNX channels-first, batch included)
    in_shape = input_shapes[0]

    # The channel axis is 1 in ONNX channels-first [N, C, H, W]
    axes_list = list(axes)
    chan_idx = axes_list.index(1) if 1 in axes_list else 0
    start_chan = int(starts[chan_idx])
    end_chan   = int(ends[chan_idx])

    n_chan_in = in_shape[1]
    # ONNX uses INT64_MAX as a sentinel for "slice to the end"; clamp to actual size
    end_chan = min(end_chan, n_chan_in)
    layer['n_chan_in']  = n_chan_in
    layer['n_chan_out'] = end_chan - start_chan
    layer['start_chan'] = start_chan
    layer['in_height']  = in_shape[2]
    layer['in_width']   = in_shape[3]

    return layer


@onnx_handler('Unsqueeze')
def parse_unsqueeze_layer(node, input_names, input_shapes, graph):
    layer = {}
    layer['name']       = node.name
    layer['class_name'] = 'Reshape'
    layer['inputs']     = [input_names[0]]
    layer['outputs']    = list(node.output)

    # opset 11: axes is an attribute; opset 13+: axes is the 2nd input tensor
    axes = get_onnx_attribute(node, 'axes')
    if axes is None and len(node.input) > 1:
        axes = list(_get_slice_constant(graph, node.input[1]).astype(int).flat)

    in_shape = list(input_shapes[0])        # [N, d1, d2, ...]
    n_out    = len(in_shape) + len(axes)    # total output dims (including batch)

    out_full = list(in_shape)
    for ax in sorted(int(a) % n_out for a in axes):
        out_full.insert(ax, 1)

    layer['target_shape'] = out_full[1:]    # exclude batch
    return layer


@onnx_handler('Squeeze')
def parse_squeeze_layer(node, input_names, input_shapes, graph):
    layer = {}
    layer['name']       = node.name
    layer['class_name'] = 'Reshape'
    layer['inputs']     = [input_names[0]]
    layer['outputs']    = list(node.output)

    in_shape = list(input_shapes[0])        # [N, d1, d2, ...]

    # opset 11: axes is an attribute; opset 13+: optional 2nd input tensor
    axes = get_onnx_attribute(node, 'axes')
    if axes is None and len(node.input) > 1:
        try:
            axes = list(_get_slice_constant(graph, node.input[1]).astype(int).flat)
        except Exception:
            axes = None

    if axes is not None:
        axes_set  = {int(a) % len(in_shape) for a in axes}
        out_shape = [d for i, d in enumerate(in_shape) if i not in axes_set]
    else:
        out_shape = [d for d in in_shape if d != 1]

    layer['target_shape'] = out_shape[1:]   # exclude batch
    return layer


@onnx_handler('Pad')
def parse_pad_layer(node, input_names, input_shapes, graph):
    layer = {}
    layer['name'] = node.name
    layer['class_name'] = 'ZeroPadding'
    layer['inputs'] = input_names
    layer['outputs'] = list(node.output)
    layer['data_format'] = (
        'channels_last' if any(node.domain == 'qonnx.custom_op.channels_last' for node in graph.node) else 'channels_first'
    )

    mode = get_onnx_attribute(node, 'mode')
    if mode is not None and mode != 'constant':
        raise RuntimeError(f'Unsupported padding mode: {mode} in node {node.name}')

    pads = get_constant_value(graph, node.input[1])
    if len(input_names) > 2:
        const_val = get_constant_value(graph, node.input[2])
        if const_val != 0:
            raise RuntimeError(f'Only constant value of 0 supported for Pad node {node.name}, got {const_val}')

    if len(input_names) > 3:
        raise RuntimeError(f'Parsing axes input of Pad node {node.name} is not supported.')

    dim = 0
    if len(input_shapes[0]) == 3:
        dim = 1  # 2D input (batch, channels, width), will use ZeroPadding1D
        if layer['data_format'] == 'channels_first':
            _, channels, width = input_shapes[0]
            pad_left, pad_right = pads[2], pads[5]
        else:
            _, width, channels = input_shapes[0]
            pad_left, pad_right = pads[1], pads[4]
        out_width = width + pad_left + pad_right

        layer['n_chan'] = channels
        layer['in_width'] = width
        layer['out_width'] = out_width

        layer['pad_left'] = pad_left
        layer['pad_right'] = pad_right
    elif len(input_shapes[0]) == 4:
        dim = 2  # 3D input (batch, channels, height, width), will use ZeroPadding2D
        if layer['data_format'] == 'channels_first':
            _, channels, height, width = input_shapes[0]
            pad_top, pad_bottom = pads[2], pads[6]
            pad_left, pad_right = pads[3], pads[7]
        else:
            _, height, width, channels = input_shapes[0]
            pad_top, pad_bottom = pads[1], pads[5]
            pad_left, pad_right = pads[2], pads[6]
        out_height = height + pad_top + pad_bottom
        out_width = width + pad_left + pad_right

        layer['n_chan'] = channels
        layer['in_height'] = height
        layer['in_width'] = width
        layer['out_height'] = out_height
        layer['out_width'] = out_width

        layer['pad_top'] = pad_top
        layer['pad_bottom'] = pad_bottom
        layer['pad_left'] = pad_left
        layer['pad_right'] = pad_right
    else:
        raise RuntimeError(f'Unsupported input shape: {input_shapes[0]} for Pad node {node.name}')

    layer['class_name'] += str(dim) + 'D'

    return layer
