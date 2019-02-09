    
# Imports
import numpy as np
import tensorflow as tf

import lucid.modelzoo.vision_models as models
from lucid.misc.io import load, save, show
import lucid.optvis.transform as transform
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render

from lucid.optvis.param.color import to_valid_rgb

##changes in fft parametrization from lucid library to make it work for 3d
def imageCube(w, h=None,d=None, sd=None, decorrelate=True, fft=True, alpha=False):
    h = h or w
    d = d or w
    batch = 1
    channels = 4 if alpha else 3
    shape = [d , w, h, channels]
    param_f = fft_imageCube
    t = param_f(shape, sd=sd)
    rgb = to_valid_rgb(t[..., :3], decorrelate=decorrelate, sigmoid=True)
    if alpha:
        a = tf.nn.sigmoid(t[..., 3:])
        return tf.concat([rgb, a], -1)
    return rgb

def rfft3d_freqs(d, h, w):
    """Computes 2D spectrum frequencies."""

    fz = np.fft.fftfreq(d)[:,None, None]
    fy = np.fft.fftfreq(h)[:,None]
    fx = np.fft.fftfreq(w//2+1)[:]
    
    
    
    return np.sqrt(fx * fx + fy * fy + fz * fz)


def fft_imageCube(shape, sd=None, decay_power=1):
    """An image paramaterization using 2D Fourier coefficients."""

    sd = sd or 0.01
    d, h, w, ch = shape
    freqs = rfft3d_freqs(d, h, w)
    init_val_size = (2, ch) + freqs.shape
    # Create a random variable holding the actual 2D fourier coefficients
    init_val = np.random.normal(size=init_val_size, scale=sd).astype(np.float32)
    spectrum_real_imag_t = tf.Variable(init_val)
    spectrum_t = tf.complex(spectrum_real_imag_t[0], spectrum_real_imag_t[1])

    # Scale the spectrum. First normalize energy, then scale by the square-root
    # of the number of pixels to get a unitary transformation.
    # This allows to use similar leanring rates to pixel-wise optimisation.
    scale = 1.0 / np.maximum(freqs, 1.0 / max(d, w, h)) ** decay_power
    scale *= np.sqrt(d * w * h)
    scaled_spectrum_t = scale * spectrum_t
    scaled_spectrum_t = tf.concat([scaled_spectrum_t,
                                   tf.conj(tf.reverse(scaled_spectrum_t[:,:,:,:-1],[3])) ],
                                  3)
    # convert complex scaled spectrum to shape (h, w, ch) image tensor
    # needs to transpose because irfft2d returns channels first
    image_t = tf.transpose(tf.real(tf.spectral.ifft3d(scaled_spectrum_t)), (1, 2, 3, 0))

    # in case of odd spatial input dimensions we need to crop
    image_t = image_t[:d, :h, :w, :ch]
    
    batched_image_t = image_t / 4.0  # TODO: is that a magic constant?
    return batched_image_t



print('loading model')

model = models.InceptionV1()
model.load_graphdef()
print('calculating')
neuron = ("mixed4a_pre_relu", 476)
version=7
size = 64 #resulting image cube haa dimensions size X size X size X 3
def param_f3d(size):
  temp = imageCube(size)
  return tf.concat([temp,
                    tf.transpose(temp,[1,0,2,3]),
                    tf.transpose(temp, [2,1,0,3])], 0)

objective = objectives.channel(*neuron)
image_cube = render.render_vis(model, objective,
                               lambda : param_f3d(size),
                               transforms= transform.standard_transforms,
                               thresholds=(512,))# threshold number of steps.
                                                      #I used 4096
  
image_cube = np.array(image_cube)[:,:size] #image cube
np.save(f"featureCube{size}_{version}.npy", image_cube)

  
