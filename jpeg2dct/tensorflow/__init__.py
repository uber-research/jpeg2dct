# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader
import sysconfig

# make sure common library is loaded
import jpeg2dct.common


def get_ext_suffix():
    """Determine library extension for various versions of Python."""
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix:
        return ext_suffix

    ext_suffix = sysconfig.get_config_var('SO')
    if ext_suffix:
        return ext_suffix

    return '.so'


def _load_library(name, op_list=None):
    """Loads a .so file containing the specified operators.

    Args:
      name: The name of the .so file to load.
      op_list: A list of names of operators that the library should have. If None
          then the .so file's contents will not be verified.

    Raises:
      NameError if one of the required ops is missing.
      NotFoundError if were not able to load .so file.
    """
    filename = resource_loader.get_path_to_datafile(name)
    library = load_library.load_op_library(filename)
    for expected_op in (op_list or []):
        for lib_op in library.OP_LIST.op:
            if lib_op.name == expected_op:
                break
        else:
            raise NameError(
                'Could not find operator %s in dynamic library %s' %
                (expected_op, name))
    return library


TF_LIB = _load_library('tf_lib' + get_ext_suffix(), ['DecodeJpeg2dct'])


def decode(buffer, normalized=True, channels=3, name=None):
    """
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
    """
    return TF_LIB.decode_jpeg2dct(buffer, normalized=normalized, channels=channels, name=name)


ops.NotDifferentiable('DecodeJpeg2dct')
