//Copyright (c) 2018 Uber Technologies, Inc.
//
//Licensed under the Uber Non-Commercial License (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at the root directory of this project.
//
//See the License for the specific language governing permissions and
//limitations under the License.

#ifndef DCTFROMJPG_H_
#define DCTFROMJPG_H_

namespace jpeg2dct {
namespace common {

struct band_info {
  short *dct;
  unsigned int dct_h;
  unsigned int dct_w;
  unsigned int dct_b;
};

void read_dct_coefficients_from_file_(char *filename, bool normalized,
                                      int channels, band_info *band1,
                                      band_info *band2, band_info *band3);

void read_dct_coefficients_from_file(
    char *filename, bool normalized, int channels, short **band1_dct,
    int *band1_dct_h, int *band1_dct_w, int *band1_dct_b, short **band2_dct,
    int *band2_dct_h, int *band2_dct_w, int *band2_dct_b, short **band3_dct,
    int *band3_dct_h, int *band3_dct_w, int *band3_dct_b);

void read_dct_coefficients_from_buffer_(char *buffer, unsigned long buffer_len,
                                        bool normalized, int channels,
                                        band_info *band1, band_info *band2,
                                        band_info *band3);

void read_dct_coefficients_from_buffer(
    char *buffer, unsigned long buffer_len, bool normalized, int channels,
    short **band1_dct, int *band1_dct_h, int *band1_dct_w, int *band1_dct_b,
    short **band2_dct, int *band2_dct_h, int *band2_dct_w, int *band2_dct_b,
    short **band3_dct, int *band3_dct_h, int *band3_dct_w, int *band3_dct_b);

} // namespace common
} // namespace jpeg2dct

#endif
