// Copyright (c) 2018 Uber Technologies, Inc.
//
// Licensed under the Uber Non-Commercial License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at the root directory of this project.
//
// See the License for the specific language governing permissions and
// limitations under the License.

%module dctfromjpg_wrapper


%{
#define SWIG_FILE_WITH_INIT
#include "../common/dctfromjpg.h"
%}

%include "numpy.i"
%init %{
import_array();
%}

%begin %{
#define SWIG_PYTHON_STRICT_BYTE_CHAR
%}

%apply (short **ARGOUTVIEWM_ARRAY3, int *DIM1, int *DIM2, int *DIM3) {(short **band1_dct, int *band1_dct_h, int *band1_dct_w, int *band1_dct_b)};
%apply (short **ARGOUTVIEWM_ARRAY3, int *DIM1, int *DIM2, int *DIM3) {(short **band2_dct, int *band2_dct_h, int *band2_dct_w, int *band2_dct_b)};
%apply (short **ARGOUTVIEWM_ARRAY3, int *DIM1, int *DIM2, int *DIM3) {(short **band3_dct, int *band3_dct_h, int *band3_dct_w, int *band3_dct_b)};

%include "exception.i"
%exception {
  try {
    $action
  } catch (std::runtime_error &e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  } catch (...) {
    SWIG_exception(SWIG_RuntimeError, "unknown exception");
  }
}

%include "../common/dctfromjpg.h"
using namespace jpeg2dct::common;
