
from jpeg2dct.numpy import loads

try:
    import cv2
    from pyspark.sql.types import BinaryType
except:
    raise Exception("cv2 and pyspark package needs to be installed, for parquest related representations")


class Jpeg2DCTNumpyCodec(object):
    def __init__(self, quality=80, is_rgb=False, normalized=True, channels=3):
        """Jpeg2DCTCodec would compress images into jpeg, but decompress into DCT coefficients
        :param quality: used when using jpeg lossy compression
        :param is_rgb: boolean indicating if image comes as rgb
        """
        self._image_codec = '.jpeg'
        self._quality = quality
        self._is_rgb = is_rgb
        self._normalized = normalized
        self._channels = channels

    def encode(self, unischema_field, array):
        """Encode the image using OpenCV"""
        if unischema_field.numpy_dtype != array.dtype:
            raise ValueError("Unexpected type of {} feature, expected {}, got {}".format(
                unischema_field.name, unischema_field.numpy_dtype, array.dtype
            ))

        if not _is_compliant_shape(array.shape, unischema_field.shape):
            raise ValueError("Unexpected dimensions of {} feature, expected {}, got {}".format(
                unischema_field.name, unischema_field.shape, array.shape
            ))

        if not (len(array.shape) == 2) and not (len(array.shape) == 3 and array.shape[2] == 3):
            raise ValueError('Unexpected image dimensions. Supported dimensions are (H, W) or (H, W, 3). '
                             'Got {}'.format(array.shape))

        image_bgr_or_gray = array
        if self._is_rgb:
            # Convert RGB to BGR
            image_bgr_or_gray = array[:, :, (2, 1, 0)]

        _, contents = cv2.imencode(self._image_codec,
                                   image_bgr_or_gray,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
        return bytearray(contents)

    def decode(self, unischema_field, value):
        """read/load the dct coefficients from a string of bytes representing a jpeg image
        :param buffer: the jpg file buffer
        :param normalized: boolean. If True, dct coefficients are normalized with quantification tables. If False, no normalization is performed.
        :param channels: number of color channels for the decoded image
        """
        dct_y, dct_cb, dct_cr = loads(value, self._normalized, self._channels)
        return dct_y, dct_cb, dct_cr

    def spark_dtype(self):
        return BinaryType()


def _is_compliant_shape(a, b):
    """Compares shapes of two arguments.
    If size of a dimensions is None, this dimension size is ignored.
    Example:
        assert _is_compliant_shape((1, 2, 3), (1, 2, 3))
        assert _is_compliant_shape((1, 2, 3), (1, None, 3))
        assert not _is_compliant_shape((1, 2, 3), (1, 10, 3))
        assert not _is_compliant_shape((1, 2), (1,))
    :return: True, if the shapes are compliant
    """
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] and b[i]:
            if a[i] != b[i]:
                return False
    return True
