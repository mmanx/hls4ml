#ifndef NNET_SLICE_H_
#define NNET_SLICE_H_

#include "nnet_common.h"

namespace nnet {

struct slice_config {
    static const unsigned in_height  = 1;
    static const unsigned in_width   = 1;
    static const unsigned n_chan_in  = 1;
    static const unsigned n_chan_out = 1;
    static const unsigned start_chan = 0;
};

// Extract a contiguous range of channels from a channels-first (NCHW) 2D tensor.
// Input  layout: [n_chan_in  * in_height * in_width]  (NCHW flat)
// Output layout: [n_chan_out * in_height * in_width]  (NCHW flat)
// Copies channels [start_chan, start_chan + n_chan_out) for every spatial position.
// Note: input arrives after a Transpose(NHWC→NCHW) so data is channel-major.
template<class data_T, class res_T, typename CONFIG_T>
void channel_slice(
    data_T data[CONFIG_T::n_chan_in  * CONFIG_T::in_height * CONFIG_T::in_width],
    res_T  res [CONFIG_T::n_chan_out * CONFIG_T::in_height * CONFIG_T::in_width]
) {
    static const unsigned HW = CONFIG_T::in_height * CONFIG_T::in_width;

    SliceLoop_C:
    for (unsigned c = 0; c < CONFIG_T::n_chan_out; c++) {
        #pragma HLS PIPELINE
        SliceLoop_HW:
        for (unsigned hw = 0; hw < HW; hw++) {
            #pragma HLS UNROLL
            res[c * HW + hw] =
                data[(CONFIG_T::start_chan + c) * HW + hw];
        }
    }
}

} // namespace nnet

#endif  // NNET_SLICE_H_
