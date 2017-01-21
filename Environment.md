# How to setup

## Amazon EC2

### Nvidia driver

```
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
sudo apt-get update && sudo apt-get install -y --no-install-recommends cuda-drivers
```

### dynamic dns


```
sudo apt-get install ddclient
```

/etc/ddclient.conf
```

```

After updating the configuration, restart the service
```
sudo systemctl restart ddclient
sudo systemctl status ddclient
```
