import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import argparse


def smallest_size_at_least(height, width, resize_min):
    """Computes new shape with the smallest side equal to `smallest_side`.

    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.

    Args:
      height: an int32 scalar tensor indicating the current height.
      width: an int32 scalar tensor indicating the current width.
      resize_min: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.

    Returns:
      new_height: an int32 scalar tensor indicating the new height.
      new_width: an int32 scalar tensor indicating the new width.
    """
    resize_min = tf.cast(resize_min, tf.float32)

    # Convert to floats to make subsequent calculations go smoothly.
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    smaller_dim = tf.minimum(height, width)
    scale_ratio = resize_min / smaller_dim

    # Convert back to ints to make heights and widths that TF ops will accept.
    new_height = tf.cast(tf.round(height * scale_ratio), tf.int32)
    new_width = tf.cast(tf.round(width * scale_ratio), tf.int32)

    return new_height, new_width

def resize_image(image, height, width, method='BILINEAR'):
    """Simple wrapper around tf.resize_images.

    This is primarily to make sure we use the same `ResizeMethod` and other
    details each time.

    Args:
      image: A 3-D image `Tensor`.
      height: The target height for the resized image.
      width: The target width for the resized image.

    Returns:
      resized_image: A 3-D tensor containing the resized image. The first two
        dimensions have the shape [height, width].
    """
    resize_func = tf.image.ResizeMethod.NEAREST_NEIGHBOR if method == 'NEAREST' else tf.image.ResizeMethod.BILINEAR
    return tf.image.resize(image, [height, width], method=resize_func)
    #return tf.image.resize_images(image, [height, width], method=resize_func, align_corners=False)


def aspect_preserving_resize(image, resize_min, channels=3, method='BILINEAR'):
    """Resize images preserving the original aspect ratio.

    Args:
      image: A 3-D image `Tensor`.
      resize_min: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.

    Returns:
      resized_image: A 3-D tensor containing the resized image.
    """
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    new_height, new_width = smallest_size_at_least(height, width, resize_min)
    return resize_image(image, new_height, new_width, method)

def central_crop(image, crop_height, crop_width, channels=3):
    """Performs central crops of the given image list.

    Args:
      image: a 3-D image tensor
      crop_height: the height of the image following the crop.
      crop_width: the width of the image following the crop.

    Returns:
      3-D tensor with cropped image.
    """
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    amount_to_be_cropped_h = height - crop_height
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = width - crop_width
    crop_left = amount_to_be_cropped_w // 2
    # return tf.image.crop_to_bounding_box(image, crop_top, crop_left, crop_height, crop_width)

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(height, crop_height),
            tf.greater_equal(width, crop_width)),
        ['Crop size greater than the image size.']
    )
    with tf.control_dependencies([size_assertion]):
        if channels == 1:
            image = tf.squeeze(image)
            crop_start = [crop_top, crop_left, ]
            crop_shape = [crop_height, crop_width, ]
        elif channels >= 3:
            crop_start = [crop_top, crop_left, 0]
            crop_shape = [crop_height, crop_width, -1]

        image = tf.slice(image, crop_start, crop_shape)

    return tf.reshape(image, [crop_height, crop_width, -1])


input_path = './input_3_224_224.bin'

def generate_npy_data(img_path, normal,mean,var,inputshape, bNHWC=False):

    input_height , input_width, input_channel = inputshape[0],inputshape[1],inputshape[2]

    img_s = tf.io.gfile.GFile(img_path, 'rb').read()
    image = tf.image.decode_jpeg(img_s)
    if input_channel==1:
        image = tf.image.rgb_to_grayscale(image)

    image = tf.cast(image, tf.float32)
    image = tf.clip_by_value(image, 0., 255.)
    image = aspect_preserving_resize(image, max(input_height, input_width), input_channel)
    image = central_crop(image, input_height, input_width)
    image = tf.image.resize(image, [input_height, input_width])

    image = image / normal
    image = (image - mean) / var
    image = image.numpy()
    if not bNHWC:
        image = np.transpose(image, (2, 0, 1))

    scale = 1
    input_type = np.int8
    dtype_min = -128
    dtype_max = 127

    image = np.round(image.astype(float) * scale)
    image = np.clip(image, dtype_min, dtype_max).astype(input_type)
    image.tofile(input_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-i', '--img_path', required=True, help='img path')
    args = parser.parse_args()

    normal = 1.0
    mean = [127.5, 127.5, 127.5]
    var = [1.0, 1.0, 1.0]
    generate_npy_data(args.img_path,normal,mean,var,(224,224,3),bNHWC=False)
    print('------generate input file end-------')
