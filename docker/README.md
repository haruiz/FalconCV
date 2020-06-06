# Installing Docker

This guide was based on [nvidia NGC guide]( https://docs.nvidia.com/ngc/ngc-vgpu-setup-guide/index.html)

## Install Docker repository
```shell script
sudo apt-get install -y apt-transport-https\
 curl ca-certificates\
 software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
 "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
```
## Install Nvidia Container
This installs nvidia-docker2 and allows $USER to use Docker as root each time
```shell script
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-docker2
sudo usermod -aG docker $USER
```

## Reboot your system
```shell script
sudo reboot
```

After reboot you can test your installation
```shell script
docker run --runtime=nvidia --rm nvcr.io/nvidia/cuda:latest nvidia-smi
```
You will see something similar to this:
```shell script
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.56       Driver Version: 440.56       CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GRID V100D-32Q      On   | 00000000:02:04.0 Off |                  N/A |
| N/A   N/A    P0    N/A /  N/A |   2192MiB / 32768MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

# Enabling GPU Support for NGC Containers like falconcv
This will install the nvidia container toolkit
```shell script
sudo apt-get install -y docker nvidia-container-toolkit
```

Finally you are ready to Build and Run your FalconCV image

# Building Image

The Dockerfiles can be built from the root project directory with the following command:

```shell script
docker build -t haruiz/falconcv -f docker/Dockerfile .
```

# Run your Container

To use FalconCV and jupyter lab execute following command, it will share your $HOME to your container, also

```shell script
docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 -v $HOME:$HOME haruiz/falconcv
```

## Go to your favorite browser and assuming you are running Docker on your local machine type:

Link: [http://localhost:8888/lab](localhost:8888/lab)

