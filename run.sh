
#!/bin/bash
apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
apt-cache policy docker-ce
apt install docker-ce

docker network create agnostic-counting

## cd nodeflux_task
cd flask
docker build --tag flask-image .
docker run --rm -p 8080:8080 --network agnostic-counting --name flask flask-image

cd ..
cd streamlit
docker build --tag streamlit .
docker run  --rm -p 5000:8501 --network agnostic-counting --name web streamlit
