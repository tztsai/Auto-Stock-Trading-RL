# ID2223 HT21 Lab Session 1
Datetime: 2021-11-12 08:00-10:00

## How to execute
1. 
        sudo docker build -t id2221_lab:v0.1 .

2.    
        sudo docker run -p 8888:8888  -v REPLACE_WITH_ABSOLUTE_PATH_OF_THE_FORDER_ON_HOST_MACHINE:/lab_session1 id2221_lab:v0.1 [^1]


## More about Docker
1. [Docker Overview](https://docs.docker.com/get-started/overview/)
2. [Installing Docker on your machine](https://docs.docker.com/get-docker/)
3. [Build an image from Dockerfile](https://docs.docker.com/engine/reference/commandline/build/)
4. [Docker run reference](https://docs.docker.com/engine/reference/run/)
5. [More about the Docker image used in the first lab session](https://hub.docker.com/r/jupyter/all-spark-notebook)


[^1]: We will need to replace ``REPLACE_WITH_ABSOLUTE_PATH_OF_THE_FORDER_ON_HOST_MACHINE`` with the absolute path to the folder on the host machine that we want to share with the running Docker container. For example, it could be "/home/ubuntu/ID2223_HT21_Lab1/lab_session1".