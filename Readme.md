# oci-dreamer

This includes some notebooks and scripts to allow generation of frame sequences that can be made into video segments.  These frame
sequences are based on the originall [Google Deepdream](https://github.com/google/deepdream.git) work.  This work is included here
as a submodule.  The work is based on the original
Google [blog post](http://googleresearch.blogspot.ch/2015/06/inceptionism-going-deeper-into-neural.html),
which references several **arxiv** articles describing the underlying trained image recognition architecture as well as the
mechanisms and techniques used to invert the training - meaning that the trained architecture model is presented with a starting
image and that image modified using gradient *ascent* in order to create an input image that maximizes a selected layer.

To ensure that you have the up to date Google repository content,
execute the following `bash` command:

```bash
git submodule update --init --recursive --remote
```

## Local notebooks

The *dream.ipynb* notebook here is a slightly modified copy of
the notebook in the `deepedream` submodule.  It executes using
the same images, but uses some deeper layers for maximizing the
activations.

## Local scripts

The dream.py python script in the *src* directory is structured
to allow it to be `import`ed into other python code as well as
including command line argument handling to allow it to be used
directly from the command line.

## References

Description of the `caffe` model and parameters - the layers
and training data and mechanisms:

GoogLeNet model
https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet

The original paper describing the GoogLeNet inception
architecture:

Going Deeper with Convolutions (Inception)
https://arxiv.org/abs/1409.4842

## Docker image

The simplest way to explore and run the notebooks and scripts here is
to build the required software and dependencies into a `docker` container
and executing from that container.

The external dependencies and some starter notebooks are included
in the container.  In addition a command line ready version
of the model executions is included so that frames can be generated
for one of more model layers without rendering the intermediate
results.

### Building

There is a single *dockerfile* included here which builds the
CPU based version of caffe and provides the GoogLeNet model definition
and parameters for use.  This is the same model as in the google/deepdream
submodule.

To build the image, run the following `bash` command:

```bash
docker build -t notebook .
```

### Running Notebooks

To execute the container to allow access to the *dream.ipynb* notebook,
use the following `bash` command:

```bash
docker run -d -v $(pwd):/home/jovyan/local -p 2112:8888 --name notebook notebook
```

This will start a `jupyter` notebook server with the local directory
mounted in the `jupyter` directory named `local`.  When stopping or
killing a notebook container, remember to remove it from the docker
engine if you want to restart another container with the same name.

Access the notebook by pointing you browser at *http://localhost:2112/login* and
entering the value `demo` when asked for a passowrd or token.  This
will give you access to the installed notebooks and images as well
as the mounted directories on the host.  If you want to save the
results of your notebook exploration beyond the current session, you
will want to create / copy your notebooks in / to the *local* subdirectory
which is mounted from the host so that any saved results will be
retained after the notebook and docker execution have completed.

### Running Command Line

The container includes the *dream.py* script in the */app*
directory of the container.  This script should be executed
as a python script using the python command from within the
container, where all required dependencies are already
present.  No modifications to the host machine are needed.

You can execute the *dream.py* script from the command line by
executing the following `bash` command:
 
```bash
docker run -v $(pwd):/home/jovyan/local --name dream-generator oci-dream-nb python /app/dream.py --image local/images/sky1024px.jpg -o local/frames
```

Note that the local direcory is mounted in the container at
*/home/jovyan/local* which is a directory within the starting
work directory of the container.  To retain any updates of output
from execution, all of the output data should be directed to
within this *local* subdirectory.

This holds for input data as well, so referencing image files
that are not present in the built container should be done via
this path as well.  The command line above illustrates these
principals.

---

To start at an intermediate location in the frame generation,
you need to specify the starting image and the starting frame
number.  Note that the script will only work correctly if these
starting points are at the beginning of a model layer.  That is
the start frame should be a multiple of 100 and the starting image
should be the final image from the previous layer.  So, for
example, this command will restart processing from the end
of the 8th layer and start building frames from the 9th layer:
 
```bash
docker run -v $(pwd):/home/jovyan/local --name dream-generator oci-dream-nb python /app/dream.py --image local/frames/00799.jpg -o local/frames --start 800
```