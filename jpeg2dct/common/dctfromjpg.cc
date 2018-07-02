//Copyright (c) 2018 Uber Technologies, Inc.
//
//Licensed under the Uber Non-Commercial License (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at the root directory of this project.
//
//See the License for the specific language governing permissions and
//limitations under the License.

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <stdio.h>

// Has to come after `stdio.h`
#include <jpeglib.h>

#include "dctfromjpg.h"

namespace jpeg2dct {
namespace common {

void unpack_band_info(band_info band, short **band_dct, int *band_dct_h,
                      int *band_dct_w, int *band_dct_b) {
  *band_dct = band.dct;
  *band_dct_h = band.dct_h;
  *band_dct_w = band.dct_w;
  *band_dct_b = band.dct_b;
}

void dummy_dct_coefficients(band_info *band) {
  band->dct_h = 0;
  band->dct_w = 0;
  band->dct_b = 0;
  band->dct = new short[0];
}

void read_dct_coefficients(jpeg_decompress_struct *srcinfo,
                           jvirt_barray_ptr *src_coef_arrays, int compNum,
                           band_info *band, bool normalized) {
  if (compNum >= srcinfo->num_components) {
    // make an empty component which would be half size of chroma
    band->dct_h = (srcinfo->comp_info[0].height_in_blocks + 1) / 2;
    band->dct_w = (srcinfo->comp_info[0].width_in_blocks + 1) / 2;
    band->dct_b = DCTSIZE2;
    long nb_elements =
        (long)(band->dct_h) * (long)(band->dct_w) * (long)(band->dct_b);
    band->dct = new short[nb_elements];
    std::memset((void *)band->dct, 0, sizeof(short) * nb_elements);
    return;
  }

  // prepare memory space dimensions
  band->dct_h = srcinfo->comp_info[compNum].height_in_blocks;
  band->dct_w = srcinfo->comp_info[compNum].width_in_blocks;
  band->dct_b = DCTSIZE2;
  long nb_elements =
      (long)(band->dct_h) * (long)(band->dct_w) * (long)(band->dct_b);
  band->dct = new short[nb_elements];

  int quant_idx = srcinfo->comp_info[compNum].quant_tbl_no;
  short unscale = 1;

  short *current_dct_coeff = band->dct;
  for (JDIMENSION rowNum = 0; rowNum < band->dct_h; rowNum++) {
    JBLOCKARRAY rowPtrs = (srcinfo->mem->access_virt_barray)(
        (j_common_ptr)srcinfo, src_coef_arrays[compNum], rowNum, (JDIMENSION)1,
        FALSE);
    for (JDIMENSION colNum = 0; colNum < band->dct_w; colNum++) {
      for (JDIMENSION c = 0; c < band->dct_b; c++) {
        if (normalized) {
          unscale = srcinfo->quant_tbl_ptrs[quant_idx]->quantval[c];
        }
        *current_dct_coeff = rowPtrs[0][colNum][c] * unscale;
        current_dct_coeff++;
      }
    }
  }
}

void add_error_handler(jpeg_error_mgr *jerr) {
  jerr->error_exit = [](j_common_ptr cinfo) {
    char pszErr[1024];
    (cinfo->err->format_message)(cinfo, pszErr);
    throw std::runtime_error(pszErr);
  };
}

void transcode(jpeg_decompress_struct *srcinfo, unsigned char **outbuffer) {
  static bool warning_emitted = false;
  if (!warning_emitted) {
    fprintf(stderr, "WARNING: Non-standard JPEG image encountered, transcoding "
                    "to H2_V2 which may negatively impact performance. This is "
                    "the only time this warning will be shown.\n");
    warning_emitted = true;
  }

  // start decompress
  (void)jpeg_start_decompress(srcinfo);

  // create the compression structure
  jpeg_compress_struct dstinfo;
  jpeg_error_mgr jerr;
  dstinfo.err = jpeg_std_error(&jerr);
  add_error_handler(&jerr);
  jpeg_create_compress(&dstinfo);

  unsigned long outlen = 0;
  jpeg_mem_dest(&dstinfo, outbuffer, &outlen);

  dstinfo.image_width = srcinfo->image_width;
  dstinfo.image_height = srcinfo->image_height;
  dstinfo.input_components = srcinfo->output_components;
  dstinfo.in_color_space = srcinfo->out_color_space;

  jpeg_set_defaults(&dstinfo);
  jpeg_set_quality(&dstinfo, 100, TRUE);
  jpeg_start_compress(&dstinfo, TRUE);

  // transcode
  unsigned char *line_buffer =
      new unsigned char[srcinfo->output_width * srcinfo->output_components];
  while (srcinfo->output_scanline < srcinfo->output_height) {
    jpeg_read_scanlines(srcinfo, &line_buffer, 1);
    (void)jpeg_write_scanlines(&dstinfo, &line_buffer, 1);
  }
  delete[] line_buffer;
  jpeg_finish_compress(&dstinfo);
  jpeg_destroy_compress(&dstinfo);

  // re-create decompress
  jpeg_destroy_decompress(srcinfo);
  jpeg_create_decompress(srcinfo);
  jpeg_mem_src(srcinfo, *outbuffer, outlen);
  (void)jpeg_read_header(srcinfo, TRUE);
}

bool is_grayscale(jpeg_decompress_struct *srcinfo) {
  return srcinfo->num_components == 1;
}

bool is_h2_v2(jpeg_decompress_struct *srcinfo) {
  return (srcinfo->comp_info[0].h_samp_factor == 2) &&
         (srcinfo->comp_info[0].v_samp_factor == 2) &&
         (srcinfo->comp_info[1].h_samp_factor == 1) &&
         (srcinfo->comp_info[1].v_samp_factor == 1) &&
         (srcinfo->comp_info[2].h_samp_factor == 1) &&
         (srcinfo->comp_info[2].v_samp_factor == 1);
}

void read_dct_coefficients_from_srcinfo(jpeg_decompress_struct *srcinfo,
                                        bool normalized, int channels,
                                        band_info *band1, band_info *band2,
                                        band_info *band3) {
  (void)jpeg_read_header(srcinfo, TRUE);

  unsigned char *buffer = nullptr;
  if (!is_grayscale(srcinfo) && !is_h2_v2(srcinfo)) {
    transcode(srcinfo, &buffer);
  }

  jvirt_barray_ptr *src_coef_arrays = jpeg_read_coefficients(srcinfo);
  read_dct_coefficients(srcinfo, src_coef_arrays, 0, band1, normalized);
  if (channels == 3) {
    read_dct_coefficients(srcinfo, src_coef_arrays, 1, band2, normalized);
    read_dct_coefficients(srcinfo, src_coef_arrays, 2, band3, normalized);
  } else {
    dummy_dct_coefficients(band2);
    dummy_dct_coefficients(band3);
  }

  if (buffer != nullptr) {
    delete[] buffer;
  }
}

void read_dct_coefficients_from_buffer_(char *jpg_buffer,
                                        unsigned long buffer_len,
                                        bool normalized, int channels,
                                        band_info *band1, band_info *band2,
                                        band_info *band3) {
  jpeg_decompress_struct srcinfo;
  jpeg_error_mgr jerr;
  srcinfo.err = jpeg_std_error(&jerr);
  add_error_handler(&jerr);
  jpeg_create_decompress(&srcinfo);

  unsigned char *u_jpg_buffer = reinterpret_cast<unsigned char *>(jpg_buffer);
  jpeg_mem_src(&srcinfo, u_jpg_buffer, buffer_len);

  read_dct_coefficients_from_srcinfo(&srcinfo, normalized, channels, band1,
                                     band2, band3);

  jpeg_destroy_decompress(&srcinfo);
}

void read_dct_coefficients_from_buffer(
    char *jpg_buffer, unsigned long buffer_len, bool normalized, int channels,
    short **band1_dct, int *band1_dct_h, int *band1_dct_w, int *band1_dct_b,
    short **band2_dct, int *band2_dct_h, int *band2_dct_w, int *band2_dct_b,
    short **band3_dct, int *band3_dct_h, int *band3_dct_w, int *band3_dct_b) {
  band_info band1, band2, band3;
  read_dct_coefficients_from_buffer_(jpg_buffer, buffer_len, normalized,
                                     channels, &band1, &band2, &band3);
  unpack_band_info(band1, band1_dct, band1_dct_h, band1_dct_w, band1_dct_b);
  unpack_band_info(band2, band2_dct, band2_dct_h, band2_dct_w, band2_dct_b);
  unpack_band_info(band3, band3_dct, band3_dct_h, band3_dct_w, band3_dct_b);
}

void read_dct_coefficients_from_file_(char *filename, bool normalized,
                                      int channels, band_info *band1,
                                      band_info *band2, band_info *band3) {
  FILE *infile;
  if ((infile = fopen(filename, "rb")) == nullptr) {
    fprintf(stderr, "ERROR: can't open %s\n", filename);
    return;
  }

  jpeg_decompress_struct srcinfo;
  jpeg_error_mgr jerr;
  srcinfo.err = jpeg_std_error(&jerr);
  add_error_handler(&jerr);
  jpeg_create_decompress(&srcinfo);
  jpeg_stdio_src(&srcinfo, infile);

  read_dct_coefficients_from_srcinfo(&srcinfo, normalized, channels, band1,
                                     band2, band3);

  jpeg_destroy_decompress(&srcinfo);
  fclose(infile);
}

void read_dct_coefficients_from_file(
    char *filename, bool normalized, int channels, short **band1_dct,
    int *band1_dct_h, int *band1_dct_w, int *band1_dct_b, short **band2_dct,
    int *band2_dct_h, int *band2_dct_w, int *band2_dct_b, short **band3_dct,
    int *band3_dct_h, int *band3_dct_w, int *band3_dct_b) {
  band_info band1, band2, band3;
  read_dct_coefficients_from_file_(filename, normalized, channels, &band1,
                                   &band2, &band3);
  unpack_band_info(band1, band1_dct, band1_dct_h, band1_dct_w, band1_dct_b);
  unpack_band_info(band2, band2_dct, band2_dct_h, band2_dct_w, band2_dct_b);
  unpack_band_info(band3, band3_dct, band3_dct_h, band3_dct_w, band3_dct_b);
}

} // namespace common
} // namespace jpeg2dct
