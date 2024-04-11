"""
This module provides functionality to build and run a Docker container
for serving machine learning models using MLflow.
"""

import os
import subprocess
import sys
import warnings
import mlflow
import click
import src


warnings.filterwarnings("ignore")
logger = src.utils.logger.setup_logger(__name__)


class Runner:
    # pylint: disable=R0903
    """
    A class to manage the build and deployment of Docker containers for serving ML models
    using MLflow.

    Attributes:
        _image_name (str): The name of the Docker image to be used.
        _experiment_name (str): The name of the MLflow experiment.
        _container_name (str): The name of the Docker container.
        _port (int): The port number on which the ML model should be served.
        _project_dir (str): Directory path of the project.
    """

    def __init__(self, image_name, experiment_name, port, project_dir):
        """
        Initializes the Runner with the necessary details for the Docker image, MLflow experiment,
        and model serving.
        """

        self._image_name = image_name
        self._experiment_name = experiment_name
        self._container_name = f"{self._image_name}_model_serving"
        self._port = port
        self._project_dir = project_dir

    def _build_image(self):
        """
        Builds the Docker image with the specified image name from the Dockerfile
        present in the current directory.
        """

        logger.info("Building Docker image")
        build_command = f"docker build -t {self._image_name} ."
        subprocess.run(build_command, shell=True, check=True)

    def _get_model_uri(self):
        """
        Fetches the best model URI from MLflow based on the experiment name provided.
        """

        mlflow.set_experiment(self._experiment_name)

        best_run_df = mlflow.search_runs(order_by=["metrics.f1 ASC"], max_results=1)
        if len(best_run_df.index) == 0:
            # pylint: disable=W0719
            raise Exception(f"Found no runs for experiment '{self._experiment_name}'")

        best_run = mlflow.get_run(best_run_df.at[0, "run_id"])
        best_model_uri = f"{best_run.info.artifact_uri}/model"

        logger.info("Best run info:")
        logger.info("\tRun id: %s", best_run.info.run_id)
        logger.info("\tRun parameters: %s", best_run.data.params)
        logger.info("\tRun score: F1 = %.4f", best_run.data.metrics["f1"])
        logger.info("\tRun model URI: %s", best_model_uri)

        return best_model_uri

    def _remove_docker_container(self):
        """
        Removes any existing Docker containers with the name specified in `_container_name`.
        """

        subprocess.run(
            f"docker rm --force {self._container_name}",
            shell=True,
            check=False,
            stdout=subprocess.DEVNULL,
        )

    def _run_container(self, best_model_uri):
        """
        Runs a Docker container with the model specified by `best_model_uri`,
        publishing the model on a specified port.

        :param best_model_uri: The URI to the MLflow model to be served in the container.
        """

        _docker_run_cmd = f"""
        docker run
        --name={self._container_name}
        --volume={self._project_dir}:{self._project_dir}
        --publish {self._port}:{self._port}
        --interactive
        --rm
        mlflow_example
        mlflow models serve --model-uri {best_model_uri} --host 0.0.0.0 --port {self._port} --workers 2 --no-conda
        """.replace(
            "\n", " "
        ).strip()

        logger.info("Running command:\n%s", _docker_run_cmd)

        subprocess.run(_docker_run_cmd, shell=True, check=True)

    def run(self):
        """
        Coordinates the steps to build the image, fetch the model URI, remove any old containers,
        and run a new container.
        """
        self._build_image()
        uri = self._get_model_uri()
        self._remove_docker_container()
        self._run_container(best_model_uri=uri)


@click.command()
@click.option(
    "--image-name",
    help="Docker Image name.",
    type=str,
    default="titanic",
    show_default=True,
)
@click.option(
    "--experiment-name",
    help="Experiment name.",
    type=str,
    default="Titanic",
    show_default=True,
)
@click.option(
    "--port", help="Endpoint port.", type=int, default=5001, show_default=True
)
def main(image_name, experiment_name, port):
    """
    Command-line entry-point to set up and execute the container runner.

    :param image_name: The name of the Docker image to build.
    :param experiment_name: The name of the MLflow experiment to fetch model from.
    :param port: The port to expose the ML model serving on.
    """

    project_dir = sys.path[0]
    os.chdir(project_dir)

    runner = Runner(
        image_name=image_name,
        experiment_name=experiment_name,
        port=port,
        project_dir=project_dir,
    )

    runner.run()


if __name__ == "__main__":
    main()  # pylint: disable=E1120
