
#!/bin/bash

# docker network create agnostic-counting

## cd nodeflux_task
cd flask
#docker build --tag flask-image .
 docker run  -p 8080:8080 --network agnostic-counting --name flask flask-image
#docker run -p 8080:8080 --name flask flask-image
docker stop flask && docker start flask

cd ..
cd streamlit
docker build --tag streamlit .
 docker run  --rm -p 5000:5000 --network agnostic-counting --name web streamlit
#docker run  --rm -p 5000:8501  --name web streamlit
