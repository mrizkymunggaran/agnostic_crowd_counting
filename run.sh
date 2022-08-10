
#!/bin/bash
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common 
curl -fsSL https://yum.dockerproject.org/gpg | sudo apt-key add - 
sudo add-apt-repository \
    "deb https://apt.dockerproject.org/repo/ \
    ubuntu-$(lsb_release -cs) \
    main" 
sudo apt-get update
sudo apt-get -y install docker-engine 
# add current user to docker group so there is no need to use sudo when running docker
sudo usermod -aG docker $(whoami)

docker network create agnostic-counting

## cd nodeflux_task
cd flask
docker build --tag flask-image .
docker run --rm -p 8080:8080 --network agnostic-counting --name flask2 flask-image

cd ..
cd streamlit
docker build --tag streamlit .
docker run  --rm -p 5000:8501 --network agnostic-counting --name web streamlit
