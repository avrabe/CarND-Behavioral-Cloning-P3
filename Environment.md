# How to setup

## Amazon EC2

I choose a 30GB Ubuntu 16.04 Server Instance

### Nvidia driver

```
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
sudo apt-get update && sudo apt-get install -y --no-install-recommends cuda-8-0
```

### dynamic dns


```
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ddclient
```

Afterwards th
/etc/ddclient.conf
```
use=web, web='http://ip1.dynupdate.no-ip.com:8245/'
daemon=5m
protocol=noip, login='<username>', password='<password>'
<domain>
```

After updating the configuration, restart the service
```
sudo systemctl restart ddclient
sudo systemctl status ddclient
```

### Miniconda

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
```

### Setup the repository

```
mkdir -p ~/git
cd ~/git
git clone https://github.com/avrabe/CarND-Behavioral-Cloning-P3.git
cd CarND-Behavioral-Cloning-P3
~/miniconda3/bin/conda env create -f environment-gpu.yml
```

And to activate the environment

```
source ~/miniconda3/bin/activate carnd-term1
```

### Getting the sample data

```
sudo apt-get install -y unzip
mkdir -p ~/data
cd ~/data
wget https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip
unzip data.zip
```

## Starting the training

```
source ~/miniconda3/bin/activate carnd-term1
cd ~/git/CarND-Behavioral-Cloning-P3
python model.py --help
```
