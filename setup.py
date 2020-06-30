# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import os
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils import ccompiler, sysconfig
from distutils.errors import CompileError, DistutilsPlatformError, LinkError
import sys
import textwrap
import traceback

from jpeg2dct import __version__

common_lib = Extension('jpeg2dct.common.common_lib', [])
numpy_lib = Extension('jpeg2dct.numpy._dctfromjpg_wrapper', [])
tf_lib = Extension('jpeg2dct.tensorflow.tf_lib', [])

JPEG_ROOT = None

DEBUG = False

def check_tf_version():
    try:
        import tensorflow as tf
        if tf.__version__ < '1.1.0':
            raise DistutilsPlatformError(
                'Your TensorFlow version %s is outdated.  '
                'Horovod requires tensorflow>=1.1.0' % tf.__version__)
    except ImportError:
        raise DistutilsPlatformError(
            'import tensorflow failed, is it installed?\n\n%s' % traceback.format_exc())
    except AttributeError:
        # This means that tf.__version__ was not exposed, which makes it *REALLY* old.
        raise DistutilsPlatformError(
            'Your TensorFlow version is outdated.  Horovod requires tensorflow>=1.1.0')


def get_cpp_flags(build_ext):
    last_err = None
    default_flags = ['-std=c++11', '-fPIC', '-O2']
    if sys.platform == 'darwin':
        # Darwin most likely will have Clang, which has libc++.
        flags_to_try = [default_flags + ['-stdlib=libc++'], default_flags]
    else:
        flags_to_try = [default_flags, default_flags + ['-stdlib=libc++']]
    for cpp_flags in flags_to_try:
        try:
            test_compile(build_ext, 'test_cpp_flags', extra_preargs=cpp_flags,
                         code=textwrap.dedent('''\
                    #include <unordered_map>
                    void test() {
                    }
                    '''))

            return cpp_flags
        except (CompileError, LinkError):
            last_err = 'Unable to determine C++ compilation flags (see error above).'
        except Exception:
            last_err = 'Unable to determine C++ compilation flags.  ' \
                       'Last error:\n\n%s' % traceback.format_exc()

    raise DistutilsPlatformError(last_err)


def get_tf_include_dirs():
    import tensorflow as tf
    tf_inc = tf.sysconfig.get_include()
    return [tf_inc, '%s/external/nsync/public' % tf_inc]


def get_tf_lib_dirs():
    import tensorflow as tf
    tf_lib = tf.sysconfig.get_lib()
    return [tf_lib]


def get_tf_libs(build_ext, lib_dirs, cpp_flags):
    last_err = None
    for tf_libs in [['tensorflow_framework'], []]:
        try:
            lib_file = test_compile(build_ext, 'test_tensorflow_libs',
                                    library_dirs=lib_dirs, libraries=tf_libs,
                                    extra_preargs=cpp_flags,
                                    code=textwrap.dedent('''\
                    void test() {
                    }
                    '''))

            from tensorflow.python.framework import load_library
            load_library.load_op_library(lib_file)

            return tf_libs
        except (CompileError, LinkError):
            last_err = 'Unable to determine -l link flags to use with TensorFlow (see error above).'
        except Exception:
            last_err = 'Unable to determine -l link flags to use with TensorFlow.  ' \
                       'Last error:\n\n%s' % traceback.format_exc()

    raise DistutilsPlatformError(last_err)


def get_tf_abi(build_ext, include_dirs, lib_dirs, libs, cpp_flags):
    last_err = None
    cxx11_abi_macro = '_GLIBCXX_USE_CXX11_ABI'
    for cxx11_abi in ['0', '1']:
        try:
            lib_file = test_compile(build_ext, 'test_tensorflow_abi',
                                    macros=[(cxx11_abi_macro, cxx11_abi)],
                                    include_dirs=include_dirs, library_dirs=lib_dirs,
                                    libraries=libs, extra_preargs=cpp_flags,
                                    code=textwrap.dedent('''\
                #include <string>
                #include "tensorflow/core/framework/op.h"
                #include "tensorflow/core/framework/op_kernel.h"
                #include "tensorflow/core/framework/shape_inference.h"
                void test() {
                    auto ignore = tensorflow::strings::StrCat("a", "b");
                }
                '''))

            from tensorflow.python.framework import load_library
            load_library.load_op_library(lib_file)

            return cxx11_abi_macro, cxx11_abi
        except (CompileError, LinkError):
            last_err = 'Unable to determine CXX11 ABI to use with TensorFlow (see error above).'
        except Exception:
            last_err = 'Unable to determine CXX11 ABI to use with TensorFlow.  ' \
                       'Last error:\n\n%s' % traceback.format_exc()

    raise DistutilsPlatformError(last_err)


def get_tf_flags(build_ext, cpp_flags):
    import tensorflow as tf
    try:
        return tf.sysconfig.get_compile_flags(), tf.sysconfig.get_link_flags()
    except AttributeError:
        # fallback to the previous logic
        tf_include_dirs = get_tf_include_dirs()
        tf_lib_dirs = get_tf_lib_dirs()
        tf_libs = get_tf_libs(build_ext, tf_lib_dirs, cpp_flags)
        tf_abi = get_tf_abi(build_ext, tf_include_dirs,
                            tf_lib_dirs, tf_libs, cpp_flags)

        compile_flags = []
        for include_dir in tf_include_dirs:
            compile_flags.append('-I%s' % include_dir)
        if tf_abi:
            compile_flags.append('-D%s=%s' % tf_abi)

        link_flags = []
        for lib_dir in tf_lib_dirs:
            link_flags.append('-L%s' % lib_dir)
        for lib in tf_libs:
            link_flags.append('-l%s' % lib)

        return compile_flags, link_flags


def test_compile(build_ext, name, code, libraries=None, include_dirs=None, library_dirs=None, macros=None,
                 extra_preargs=None):
    test_compile_dir = os.path.join(build_ext.build_temp, 'test_compile')
    if not os.path.exists(test_compile_dir):
        os.makedirs(test_compile_dir)

    source_file = os.path.join(test_compile_dir, '%s.cc' % name)
    with open(source_file, 'w') as f:
        f.write(code)

    compiler = build_ext.compiler
    [object_file] = compiler.object_filenames([source_file])
    shared_object_file = compiler.shared_object_filename(
        name, output_dir=test_compile_dir)

    compiler.compile([source_file], extra_preargs=extra_preargs,
                     include_dirs=include_dirs, macros=macros)
    compiler.link_shared_object(
        [object_file], shared_object_file, libraries=libraries, library_dirs=library_dirs)

    return shared_object_file


def get_conda_include_dir():
    prefix = os.environ.get('CONDA_PREFIX', '.')
    return [os.path.join(prefix,'include')]

def _dbg(s, tp=None):
    if DEBUG:
        if tp:
            print(s % tp)
            return
        print(s)
        
def _cmd_exists(cmd):
    return any(
        os.access(os.path.join(path, cmd), os.X_OK)
        for path in os.environ["PATH"].split(os.pathsep)
    )

def _pkg_config(name):
    try:
        command = os.environ.get("PKG_CONFIG", "pkg-config")
        command_libs = [command, "--libs-only-L", name]
        command_cflags = [command, "--cflags-only-I", name]
        if not DEBUG:
            command_libs.append("--silence-errors")
            command_cflags.append("--silence-errors")
        libs = (
            subprocess.check_output(command_libs)
            .decode("utf8")
            .strip()
            .replace("-L", "")
        )
        cflags = (
            subprocess.check_output(command_cflags)
            .decode("utf8")
            .strip()
            .replace("-I", "")
        )
        return (libs, cflags)
    except Exception:
        pass

def _add_directory(path, subdir, where=None):
    if subdir is None:
        return
    subdir = os.path.realpath(subdir)
    if os.path.isdir(subdir) and subdir not in path:
        if where is None:
            _dbg("Appending path %s", subdir)
            path.append(subdir)
        else:
            _dbg("Inserting path %s", subdir)
            path.insert(where, subdir)
    elif subdir in path and where is not None:
        path.remove(subdir)
        path.insert(where, subdir)

def get_lib_and_include():
    library_dirs = []
    include_dirs = []

    _add_directory(include_dirs, "src/libImaging")

    pkg_config = None
    if _cmd_exists(os.environ.get("PKG_CONFIG", "pkg-config")):
        pkg_config = _pkg_config

    #
    # add configured kits
    for root_name, lib_name in dict(
        JPEG_ROOT="libjpeg"
    ).items():
        root = globals()[root_name]

        if root is None and root_name in os.environ:
            prefix = os.environ[root_name]
            root = (os.path.join(prefix, "lib"), os.path.join(prefix, "include"))

        if root is None and pkg_config:
            if isinstance(lib_name, tuple):
                for lib_name2 in lib_name:
                    _dbg("Looking for `%s` using pkg-config." % lib_name2)
                    root = pkg_config(lib_name2)
                    if root:
                        break
            else:
                _dbg("Looking for `%s` using pkg-config." % lib_name)
                root = pkg_config(lib_name)

        if isinstance(root, tuple):
            lib_root, include_root = root
        else:
            lib_root = include_root = root

        _add_directory(library_dirs, lib_root)
        _add_directory(include_dirs, include_root)

    # respect CFLAGS/CPPFLAGS/LDFLAGS
    for k in ("CFLAGS", "CPPFLAGS", "LDFLAGS"):
        if k in os.environ:
            for match in re.finditer(r"-I([^\s]+)", os.environ[k]):
                _add_directory(include_dirs, match.group(1))
            for match in re.finditer(r"-L([^\s]+)", os.environ[k]):
                _add_directory(library_dirs, match.group(1))

    # include, rpath, if set as environment variables:
    for k in ("C_INCLUDE_PATH", "CPATH", "INCLUDE"):
        if k in os.environ:
            for d in os.environ[k].split(os.path.pathsep):
                _add_directory(include_dirs, d)

    for k in ("LD_RUN_PATH", "LIBRARY_PATH", "LIB"):
        if k in os.environ:
            for d in os.environ[k].split(os.path.pathsep):
                _add_directory(library_dirs, d)

    prefix = sysconfig.get_config_var("prefix")
    if prefix:
        _add_directory(library_dirs, os.path.join(prefix, "lib"))
        _add_directory(include_dirs, os.path.join(prefix, "include"))
        
    return library_dirs, include_dirs

def get_common_options(build_ext):
    cpp_flags = get_cpp_flags(build_ext)
    
    LIBRARY_DIRS, INCLUDE_DIRS = get_lib_and_include()
    
    MACROS = []
    INCLUDE_DIRS += get_conda_include_dir()
    SOURCES = []
    COMPILE_FLAGS = cpp_flags
    LINK_FLAGS = []
    LIBRARIES = []

    return dict(MACROS=MACROS,
                INCLUDE_DIRS=INCLUDE_DIRS,
                SOURCES=SOURCES,
                COMPILE_FLAGS=COMPILE_FLAGS,
                LINK_FLAGS=LINK_FLAGS,
                LIBRARY_DIRS=LIBRARY_DIRS,
                LIBRARIES=LIBRARIES)


def build_common_extension(build_ext, options, abi_compile_flags):
    common_lib.define_macros = options['MACROS']
    common_lib.include_dirs = options['INCLUDE_DIRS']
    common_lib.sources = options['SOURCES'] + ['jpeg2dct/common/dctfromjpg.cc']
    common_lib.extra_compile_args = options['COMPILE_FLAGS'] + \
                                   abi_compile_flags
    common_lib.extra_link_args = options['LINK_FLAGS']
    common_lib.library_dirs = options['LIBRARY_DIRS']
    common_lib.libraries = options['LIBRARIES'] + ["jpeg"]

    build_ext.build_extension(common_lib)


def build_numpy_extension(build_ext, options, abi_compile_flags):
    import numpy
    numpy_lib.define_macros = options['MACROS']
    numpy_lib.include_dirs = options['INCLUDE_DIRS'] + [numpy.get_include()]
    numpy_lib.sources = options['SOURCES'] + ['jpeg2dct/numpy/dctfromjpg_wrap.cc']
    numpy_lib.extra_compile_args = options['COMPILE_FLAGS'] + \
                                   abi_compile_flags
    numpy_lib.extra_link_args = options['LINK_FLAGS']
    numpy_lib.library_dirs = options['LIBRARY_DIRS']
    numpy_lib.libraries = options['LIBRARIES'] + ["jpeg"]

    build_ext.build_extension(numpy_lib)


def build_tf_extension(build_ext, options):
    check_tf_version()
    tf_compile_flags, tf_link_flags = get_tf_flags(
        build_ext, options['COMPILE_FLAGS'])

    tf_lib.define_macros = options['MACROS']
    tf_lib.include_dirs = options['INCLUDE_DIRS']
    tf_lib.sources = options['SOURCES'] + ['jpeg2dct/tensorflow/tf_lib.cc']
    tf_lib.extra_compile_args = options['COMPILE_FLAGS'] + \
        tf_compile_flags
    tf_lib.extra_link_args = options['LINK_FLAGS'] + tf_link_flags
    tf_lib.library_dirs = options['LIBRARY_DIRS']
    tf_lib.libraries = options['LIBRARIES'] + ["jpeg"]

    build_ext.build_extension(tf_lib)

    # Return ABI flags used for TensorFlow compilation.  We will use this flag
    # to compile all the libraries.
    return [flag for flag in tf_compile_flags if '_GLIBCXX_USE_CXX11_ABI' in flag]


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        options = get_common_options(self)
        abi_compile_flags = []
        built_plugins = []
        if not os.environ.get('JPEG2DCT_WITHOUT_TENSORFLOW'):
            try:
                abi_compile_flags = build_tf_extension(self, options)
                built_plugins.append(True)
            except:
                if not os.environ.get('JPEG2DCT_WITH_TENSORFLOW'):
                    print('INFO: Unable to build TensorFlow plugin, will skip it.\n\n'
                          '%s' % traceback.format_exc(), file=sys.stderr)
                    built_plugins.append(False)
                else:
                    raise
        build_common_extension(self, options, abi_compile_flags)
        build_numpy_extension(self, options, abi_compile_flags)


setup(name='jpeg2dct',
      version=__version__,
      packages=find_packages(),
      description=textwrap.dedent('''\
          Library providing a Python function and a TensorFlow Op to read JPEG image as a numpy 
          array or a Tensor containing DCT coefficients.'''),
      author='Uber Technologies, Inc.',
      long_description=textwrap.dedent('''\
          jpeg2dct library provides native Python function and a TensorFlow Op to read JPEG image
          as a numpy array or a Tensor containing DCT coefficients.'''),
      url='https://github.com/uber-research/jpeg2dct',
      ext_modules=[common_lib, numpy_lib, tf_lib],
      cmdclass={'build_ext': custom_build_ext},
      setup_requires=['numpy'],
      install_requires=['numpy'],
      tests_require=['pytest'],
      zip_safe=False)
