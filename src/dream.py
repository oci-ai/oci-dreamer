
import sys

import numpy as np
import scipy.ndimage as nd
import PIL.Image
from google.protobuf import text_format

import caffe

# If your GPU supports CUDA and Caffe was built with CUDA support,
# uncomment the following to run Caffe operations on the GPU.
# caffe.set_mode_gpu()
# caffe.set_device(0) # select GPU device if multiple devices exist

# Get the classifier model
# From the original, this is the LeNet trained on ImageNet
def getClassifier(net_fn,param_fn):
    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True
    open('tmp.prototxt', 'w').write(str(model))

    net = caffe.Classifier('tmp.prototxt', param_fn,
                           mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                           channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

    return net

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

# Objective function - L2 norm
def objective_L2(dst):
    dst.diff[:] = dst.data

# Gradient Ascent step function
def make_step(net, step_size=1.5, end='inception_4c/output',
              jitter=32, clip=True, objective=objective_L2):
    '''Basic gradient ascent step.'''

    src = net.blobs['data']  # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter + 1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2)  # apply jitter shift

    net.forward(end=end)
    objective(dst)  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size / np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2)  # unshift image

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255 - bias)

# Do the processing here
def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4,
              end='inception_4c/output', clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0 / octave_scale, 1.0 / octave_scale), order=1))

    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1])  # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0 * h / h1, 1.0 * w / w1), order=1)

        src.reshape(1, 3, h, w)  # resize the network's input image size
        src.data[0] = octave_base + detail
        for i in xrange(iter_n):
            make_step(net, end=end, clip=clip, **step_params)

        # extract details produced on the current octave
        detail = src.data[0] - octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])

# Generate a number of frames for a number of layers
def generateFrames(img,
                   net,
                   outdir='frames',
                   start_frame=0,
                   frames_per_layer=100,
                   layers=None):
    if layers is None:
        layers = net.blobs.keys()  # [1:] to skip the original image
    frame   = img
    frame_i = start_frame
    h, w = frame.shape[:2]
    s = 0.05 # scale coefficient - this is the zoom factor

    for layer in layers:
        print('Processing frame: {} as layer: {}'.format(frame_i,layer), file=sys.stderr)
        for i in xrange(frames_per_layer):
            frame = deepdream(net, frame, end=layer)
            PIL.Image.fromarray(np.uint8(frame)).save(outdir + "/%05d.jpg"%frame_i)
            frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
            frame_i += 1

def run(**kwargs):
    print('setting paths', file=sys.stderr)
    network_filename   = kwargs.get('modelpath') + 'deploy.prototxt'
    parameter_filename = kwargs.get('modelpath') + kwargs.get('parameters')

    print('getting classifier', file=sys.stderr)
    net = getClassifier( network_filename, parameter_filename )

    print('getting image', file=sys.stderr)
    img = np.float32(PIL.Image.open(kwargs.get('image')))

    print('generating', file=sys.stderr)
    frames_per_layer = 100
    first_layer      = int( int(kwargs.get('start')) / frames_per_layer)

    # Can't find the split layers from the list when running
    layers = filter(lambda x: 'split' not in x, net.blobs.keys())
    generateFrames(img,
                   net,
                   outdir=kwargs.get('outpath'),
                   start_frame=int( kwargs.get('start') ),
                   frames_per_layer=frames_per_layer,
                   layers=layers[first_layer:])

def parseArgs():
    import argparse
    parser = argparse.ArgumentParser('dream.py')
    parser.add_argument("--image",     "-i",default="images/sky1024px.jpg",
                        help="path for output image files")
    parser.add_argument("--modelpath", "-m",default="/opt/caffe/models/bvlc_googlenet/",
                        help="base path for model file location")
    parser.add_argument("--outpath",   "-o",default="frames",
                        help="path for output image files")
    parser.add_argument("--start",     "-s",default=0,
                        help="starting frame")
    parser.add_argument("--parameters","-p",default="bvlc_googlenet.caffemodel",
                        help="path for output image files")
    args = parser.parse_args()
    return vars(args)

if __name__ == "__main__":
    try:
        args = parseArgs()
        print('Args:', file=sys.stderr)
        print(args, file=sys.stderr)
        run(**args)

    except (KeyboardInterrupt, SystemExit):
        pass
    except ValueError as err:
        print("Value error: {0}".format(err), file=sys.stderr)
    except TypeError as err:
        print("Type error: {0}".format(err), file=sys.stderr)
    except IOError as err:
        print("IO error: {0}".format(err), file=sys.stderr)
    except OSError as err:
        print("OS error: {0}".format(err), file=sys.stderr)
    except:
        print("Unexpected error:", sys.exc_info()[0], file=sys.stderr)


