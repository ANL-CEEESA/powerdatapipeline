# Power Data Pipeline

This is the data pipeline  repository for **Adaptive Cybersecurity for DER: A Game-Theoretic and Machine Learning approach for Real-Time Threat Detection and Mitigation**. A high-level overview of the data pipeline is shown in Fig.1.

![high_level_overvew](..\docs\diagrams\datapreprocessing_pipeline.jpg)
Fig. 1. Data pre-processing pipeline

## Project status
**Active development**

## Repository organization
1. Please find detailed code guidelines [here](docs\Coding Standards Guide.md).

2. Please do not commit directly to the main branch. Create branches and create merge requests.

3. Please use separate folders for logically related code deliverables from tasks/sub-tasks.

## Software development tools
The table below gives information on the commonly agreed upon software solutions used in this project.

| Software requirement          | Solution                                                     |
| ----------------------------- | ------------------------------------------------------------ |
| Programming language          | Python 3.11 or higher                                        |
| Software version control      | Git                                                          |
| Machine Learning framework    | Keras 3 with TensorFlow backend |
| Container management          | Docker or Podman                                             |

## Getting started

Please clone this repository using the below steps:
```
git clone https://git.cels.anl.gov/ceser_dercybersecurity/powerdatapipeline.git
cd powerdatapipeline
```

### Test multi-source data load-extract-transform pipeline
The data pipeline will ingest CSV files from multiple sources and produce dataset objects that can be directly fed by `model.fit`. It  streams data from disk. The data pipeline is configured through the [JSON configuration file](..\config\datafusion_config_der.json).
```shell
cd examples
python3 datapipeline_test.py
```

###Pulling and commiting updates to the repository
To update local repository with changes from GitLab repository:
```
git pull origin
```
To create a new branch:

```
git checkout -b yourbranchname
```
To push the changes made in your local branch (*yourbranchname*) to the remote repository, specifically to the branch with the same name on the remote repository:

```
git push origin yourbranchname
```
To learn more about using Git, please use this [tutorial](https://docs.gitlab.com/ee/tutorials/learn_git.html).

## Docker image used for Development
Use the [**Dockerfile**](Dockerfile) to build a Docker image which can be used for running containers. Make sure that that the code added to the repository will run within the container.

### Build docker image
```
docker build -t powerdatapipeline:0.0.1 .
```
### Run Docker container using Docker image
```
docker run --rm -it -v "~/powerdatapipeline:/home/powerdatapipeline" powerdatapipeline:0.0.1

## Acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

