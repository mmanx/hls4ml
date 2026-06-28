#ifndef NNET_REDUCE_H_
#define NNET_REDUCE_H_

#include "nnet_common.h"

namespace nnet {

struct reduce_max_config {
    static const unsigned in_height = 1;
    static const unsigned in_width  = 1;
    static const unsigned n_chan    = 1;
};

// ReduceMax along the channel axis.
// Input  layout: [n_chan * in_height * in_width]  (NCHW flat)
// Output layout: [in_height * in_width]            (spatial map, channel axis removed)
// Each spatial position keeps only the maximum value across all n_chan channels.
// Note: input arrives after a Transpose(NHWC→NCHW) so data is channel-major.
template<class data_T, class res_T, typename CONFIG_T>
void reduce_max_channel(
    data_T data[CONFIG_T::n_chan * CONFIG_T::in_height * CONFIG_T::in_width],
    res_T  res [CONFIG_T::in_height * CONFIG_T::in_width]
) {
    static const unsigned HW = CONFIG_T::in_height * CONFIG_T::in_width;

    ReduceMaxLoop_HW:
    for (unsigned hw = 0; hw < HW; hw++) {
        #pragma HLS PIPELINE
        data_T max_val = data[0 * HW + hw];
        ReduceMaxLoop_C:
        for (unsigned c = 1; c < CONFIG_T::n_chan; c++) {
            #pragma HLS UNROLL
            data_T val = data[c * HW + hw];
            if (val > max_val) max_val = val;
        }
        res[hw] = max_val;
    }
}

} // namespace nnet

#endif  // NNET_REDUCE_H_
