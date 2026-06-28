from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import ChannelReduceMax

reduce_config_template = """struct config{index} : nnet::reduce_max_config {{
    static const unsigned in_height = {in_height};
    static const unsigned in_width  = {in_width};
    static const unsigned n_chan    = {n_chan};
}};\n"""

reduce_function_template = 'nnet::reduce_max_channel<{input_t}, {output_t}, config{index}>({input}, {output});'

reduce_include_list = ['nnet_utils/nnet_reduce.h']


class ChannelReduceMaxConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(ChannelReduceMax)
        self.template = reduce_config_template

    def format(self, node):
        params = self._default_config_params(node)
        return self.template.format(**params)


class ChannelReduceMaxFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(ChannelReduceMax, include_header=reduce_include_list)
        self.template = reduce_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)
