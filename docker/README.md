# Building Images

The Dockerfiles can be built from the root project directory with the following command:

```shell script
docker build -t falconcv -f docker/Dockerfile .
```

Execute as follows:

```shell script
 docker run --gpus all --rm -it -v $HOME:$HOME falconcv
```