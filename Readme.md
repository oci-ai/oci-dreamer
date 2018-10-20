# oci-dreamer

This includes some notebooks and scripts to allow generation of frame sequences that can be made into video segments.  These frame
sequences are based on the originall [Google Deepdream](https://github.com/google/deepdream.git) work.  This work is included here
as a submodule.  The work is based on the original
Google [blog post](http://googleresearch.blogspot.ch/2015/06/inceptionism-going-deeper-into-neural.html),
which references several **arxiv** articles describing the underlying trained image recognition architecture as well as the
mechanisms and techniques used to invert the training - meaning that the trained architecture model is presented with a starting
image and that image modified using gradient *ascent* in order to create an input image that maximizes a selected layer.

To ensure that you have the up to date Google repository content, you can do:

```bash
bash> git submodule update --init --recursive --remote
```

## Local libraries

## Docker image

### Building

```bash
bash> docker build -t notebook .
```

### Running

```bash
bash> docker run -d -v $(pwd):/home/jovyan/local -p 2112:8888 --name notebook notebook
```

## Local notebooks


