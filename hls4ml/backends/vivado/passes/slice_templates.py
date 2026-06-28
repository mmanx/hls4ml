from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import ChannelSlice

slice_config_template = """struct config{index} : nnet::slice_config {{
    static const unsigned in_height  = {in_height};
    static const unsigned in_width   = {in_width};
    static const unsigned n_chan_in  = {n_chan_in};
    static const unsigned n_chan_out = {n_chan_out};
    static const unsigned start_chan = {start_chan};
}};\n"""

slice_function_template = (
    'nnet::channel_slice<{input_t}, {output_t}, config{index}>({input}, {output});'
)

slice_include_list = ['nnet_utils/nnet_slice.h']


class ChannelSliceConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(ChannelSlice)
        self.template = slice_config_template

    def format(self, node):
        params = self._default_config_params(node)
        return self.template.format(**params)


class ChannelSliceFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(ChannelSlice, include_header=slice_include_list)
        self.template = slice_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)
