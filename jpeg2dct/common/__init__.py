import ctypes
import os
import sysconfig


def get_ext_suffix():
    """Determine library extension for various versions of Python."""
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix:
        return ext_suffix

    ext_suffix = sysconfig.get_config_var('SO')
    if ext_suffix:
        return ext_suffix

    return '.so'


ctypes.CDLL(os.path.join(os.path.dirname(__file__),
                         'common_lib' + get_ext_suffix()), mode=ctypes.RTLD_GLOBAL)
