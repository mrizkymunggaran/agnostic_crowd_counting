
docker network create agnostic-counting

## cd nodeflux_task
cd flask
docker build --tag flask-image .
docker run --rm -p 8080:8080 --network agnostic-counting --name flask2 flask-image

cd ..
cd streamlit
docker build --tag streamlit .
docker run  --rm -p 5000:8501 --network agnostic-counting --name web streamlit
