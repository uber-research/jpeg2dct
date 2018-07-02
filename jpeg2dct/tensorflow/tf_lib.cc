//Copyright (c) 2018 Uber Technologies, Inc.
//
//Licensed under the Uber Non-Commercial License (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at the root directory of this project.
//
//See the License for the specific language governing permissions and
//limitations under the License.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "../common/dctfromjpg.h"

using namespace tensorflow;

namespace jpeg2dct {
namespace tensorflow {

using namespace jpeg2dct::common;

class DecodeJpeg2dctOp : public OpKernel {
public:
  explicit DecodeJpeg2dctOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("normalized", &normalized_));
    OP_REQUIRES_OK(context, context->GetAttr("channels", &channels_));
  }

  void Compute(OpKernelContext *context) override {
    auto &tensor = context->input(0);
    const StringPiece input = tensor.scalar<string>()();
    band_info bands[3];
    try {
      read_dct_coefficients_from_buffer_(
          const_cast<char *>(input.data()), input.size(), normalized_,
          (int)channels_, &bands[0], &bands[1], &bands[2]);
    } catch (std::runtime_error &e) {
      context->CtxFailure(errors::Unknown(e.what()));
      return;
    }
    for (int i = 0; i < channels_; i++) {
      auto &band = bands[i];
      auto band_shape = {int64(band.dct_h), int64(band.dct_w),
                         int64(band.dct_b)};
      Tensor *band_tensor;
      auto status =
          context->allocate_output(i, TensorShape(band_shape), &band_tensor);
      std::memcpy((void *)band_tensor->tensor_data().data(),
                  (const void *)band.dct, band_tensor->tensor_data().size());
      if (!status.ok()) {
        context->CtxFailure(status);
        return;
      }
    }
  }

private:
  bool normalized_;
  int64 channels_;
};

REGISTER_KERNEL_BUILDER(Name("DecodeJpeg2dct").Device(DEVICE_CPU),
                        DecodeJpeg2dctOp);

REGISTER_OP("DecodeJpeg2dct")
    .Attr("normalized: bool = true")
    .Attr("channels: int >= 1 = 3")
    .Input("tensor: string")
    .Output("output: channels * int16")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      int64 channels;
      TF_RETURN_IF_ERROR(c->GetAttr("channels", &channels));
      if (channels != 3 && channels != 1) {
        return errors::InvalidArgument("channels should be 3 or 1");
      }
      for (int i = 0; i < channels; i++) {
        c->set_output(i, c->MakeShape({c->UnknownDim(), c->UnknownDim(), 64}));
      }
      return Status::OK();
    })
    .Doc(R"doc(
Read/load the DCT coefficients from a string of bytes representing a JPEG image.

Arguments
    buffer: the JPEG file buffer
    normalized: boolean. If True, dct coefficients are normalized with quantification tables.
                If False, no normalization is performed.
    channels: number of color channels for the decoded image.

Output
   output: (dct_y, dct_c, dct_r) as Tensors of size h x w x nb dct coef.
           given an image of size 512 x 512 x 64, the dct_y will be 64 x 64 x 64 and
           dct_c, dct_r will be 32 x 32 x 64
)doc");

} // namespace tensorflow
} // namespace jpeg2dct
