# Faster Neural Networks Straight from JPEG: jpeg2dct subroutines

This repository contains source code useful for reproducing results presented in the paper [Faster Neural Networks Straight from JPEG](https://openreview.net/forum?id=S1ry6Y1vG) (ICLR workshop 2018):

```
@inproceedings{gueguen_2018_ICLR
  title={Faster Neural Networks Straight from JPEG},
  author={Lionel Gueguen and Alex Sergeev and Ben Kadlec and Rosanne Liu and Jason Yosinski},
  booktitle={International Conference on Learning Representations},
  year={2018}
}
```

## jpeg2dct subroutines

The jpeg2dct library provides native Python functions and a TensorFlow Operators to read the Discrete Cosine Transform coefficients from image encoded in JPEG format.
The I/O operation leverages standard JPEG libraries ([libjpeg](http://libjpeg.sourceforge.net/) or [libjpeg-turbo](https://libjpeg-turbo.org/)) to perform the Huffman decoding and obtain the DCT coefficients.

## Usage
#### Read into numpy array
```python
from jpeg2dct.numpy import load, loads


#read from a file
jpeg_file = '/<jpeg2dct dir>/test/data/DCT_16_16.jpg'
dct_y, dct_cb, dct_cr = load(jpeg_file)
print ("Y component DCT shape {} and type {}".format(dct_y.shape, dct_y.dtype))
print ("Cb component DCT shape {} and type {}".format(dct_cb.shape, dct_cb.dtype))
print ("Cr component DCT shape {} and type {}".format(dct_cr.shape, dct_cr.dtype))


#read from in memory buffer
with open(jpeg_file, 'rb') as src:
    buffer = src.read()
dct_y, dct_cb, dct_cr = loads(buffer)

```
#### Read into Tensorflow Op
Example 1
```python
import tensorflow as tf
from jpeg2dct.tensorflow import batch_decode

# assemble the graph
batch_size = 4
images_byte_tensor = tf.placeholder(shape=(batch_size,), dtype=tf.string)
dcty_batched, dctc_batched, dctr_batched = batch_decode(images_byte_tensor)

# assemble the data to be fed to the graph
jpeg_file = '/<jpeg2dct dir>/test/data/DCT_16_16.jpg'
with open(jpeg_file) as src:
    byt = src.read()
images_byte_values = [byt for i in range(batch_size)]

with tf.Session() as sess:
    y, c, r = sess.run([dcty_batched, dctc_batched, dctr_batched],
                        feed_dict={images_byte_tensor:images_byte_values})

```

Example 2
```python
import tensorflow as tf
from jpeg2dct.tensorflow import decode

jpeg_file = '/<jpeg2dct dir>/test/data/DCT_16_16.jpg'
with tf.Session() as sess:
    jpegbytes = tf.read_file(jpeg_file)
    dct_y_tf, dct_c_tf, dct_r_tf = decode(jpegbytes)
    print ("Y component DCT shape {} and type {}".format(dct_y_tf.eval().shape, dct_y_tf.dtype))

```


## Installation
#### Requirements
1. Numpy>=1.14.0
2. libjpeg or libjpeg-turbo
2. (Optional) Tensorflow>=1.5.0

#### Pip
```commandline
pip install jpeg2dct
```

On macOS 10.13, with default Python, the compiler has troubles. In Conda, the following is unnecessary.
```commandline
mv /usr/local/include /usr/local/include_old
brew reinstall llvm libjpeg
pip install jpeg2dct
```

#### From source

```commandline
git clone https://github.com/uber-research/jpeg2dct.git
cd jpeg2dct
python setup.py install
```

On Mac run the following, before python setup.py ...

```commandline
export MACOSX_DEPLOYMENT_TARGET=10.10
# or
conda install --channel https://conda.anaconda.org/anaconda clangxx_osx-64
```

#### Test the installation

```commandline
python setup.py test
# or
python setup.py develop
pytest
```
