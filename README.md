# How to use BMnet+

## Docker Network Bridge

Because we will use different docker images for each service (Streamlit for web interface and Flask for API serving), we need both service to communicate. This can be done by Docker’s network bridge using this command: 

```
docker network create agnostic-counting
```

## Run the Model API Serving Docker

After we create the network bridge, we can just create both docker running by: 
```
cd flask
docker build --tag flask-image .
docker run -p 8080:8080 --network agnostic-counting --name flask flask-image
```
## Run the Streamlit Web Docker

```
cd ..
cd streamlit
docker build --tag streamlit .
docker run -p 5000:8501 --network agnostic-counting --name web streamlit
```

## Notes
Original repo : https://github.com/ahmadirfaan1/nodeflux-task
All of those commands are with the assumption that you are using one host for both docker services. If you differ the host for the services you can just change the URL on the ``streamlit/.streamlit/secrets.toml`` to your flask API URL.
