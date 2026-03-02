# ANTsPyNet docker

Container for ANTsTorch with an option to pre-install data and pre-trained networks.


## Downloading images

The container image is available on Docker Hub as `antsx/antstorch:latest`. A version with
data included is available as `antsx/antstorch:latest-with-data`.


## Building the container

From `ANTsTorch/`,

```
docker build \
  -t your_dockerhub_username/antstorch:latest \
  -f docker/Dockerfile \
  .
```

## Run time data and pretrained networks

ANTsTorch downloads data at run time. ANTsTorch code distinguishes data (eg, a
template brain) from pretrained networks, but the issues around downloading and
storing them are the same, so we'll use "data" for both types here.

By default, data is downloaded on demand and stored in a cache directory at
`${HOME}/.antstorch`. With the default user, attempts to download data at run time will fail
because the directory `/home/antstorchuser` is not writeable. This is by design, to prevent
users unknowingly downloading large amounts of data by running a container repeatedly, or
by running many containers in parallel.

To include all available data in the container image, build with the data included:

```
docker build \
    -t your_dockerhub_username/antstorch:latest \
    -f docker/Dockerfile \
    --build-arg INSTALL_ANTSTORCH_DATA=1 \
    .
```

This will make the container larger, but all data and pretrained networks will be
available at run time without downloading. You can also download a subset of data /
networks, see the help for the `tools/download_antstorch_data.py` script.


## Running the container

The docker user is `antstorchuser`, and the home directory `/home/antstorchuser` exists in the
container. The container always has the ANTsPy data, so that you can call `ants.get_data`
and run ANTsPy tests.

If you build the container with the `INSTALL_ANTSTORCH_DATA` build
arg set, the container will also have all ANTsPyNet data and pretrained networks under
`/home/antstorchuser/.antstorch`.


### Apptainer / Singularity usage

Apptainer always runs as the system user, so you will need

```
apptainer run --containall --no-home --home /home/antstorchuser antstorch_latest.sif
```

in order for ANTsPy and ANTsTorch to find built in data.

