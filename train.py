"""
This module sets up a machine learning experiment using MLflow to track and manage runs.
"""

import os
import warnings
import platform
import time
import pandas as pd
import click
import sklearn
import mlflow
import src


warnings.filterwarnings("ignore")
logger = src.utils.logger.setup_logger(__name__)


logger.info("Versions:")
logger.info("\tMLflow Version: %s", mlflow.__version__)
logger.info("\tSklearn Version: %s", sklearn.__version__)
logger.info("\tMLflow Tracking URI: %s", mlflow.get_tracking_uri())
logger.info("\tPython Version: %s", platform.python_version())
logger.info("\tOperating System: %s", platform.system() + " - " + platform.release())
logger.info("\tPlatform: %s", platform.platform())


class Trainer:
    # pylint: disable=R0903
    """
    Trainer class for setting up and running MLflow experiments.

    Attributes:
        experiment_name (str): Name of the MLflow experiment.
        data_path (str): Directory path to the dataset.
        log_plot (bool): Flag to determine whether to log plots.
    """

    def __init__(self, experiment_name, data_path, log_plot=False):
        """
        Initializes the Trainer with the specified experiment name, dataset path,
        and an option to log plots.

        :param experiment_name: The name of the MLflow experiment to be used or created.
        :param data_path: The filesystem path to the training and test datasets.
        :param log_plot: If True, generate and log plot artifacts.
        """

        self._client = mlflow.client.MlflowClient()
        self._now = src.utils.timestamp.fmt_ts_seconds(round(time.time()))
        self._col_label = "Survived"
        self.experiment_name = experiment_name
        self.data_path = data_path
        self.log_plot = log_plot

        self._pre_process = src.fit_model.FitModel()

        self._df = pd.read_csv(os.path.join(self.data_path, "train.csv"))
        self._columns = self._df.columns

        self._x_test = pd.read_csv(os.path.join(self.data_path, "test.csv"))
        self._y_test = pd.read_csv(
            os.path.join(self.data_path, "gender_submission.csv")
        )[[self._col_label]]

        if self.experiment_name:
            mlflow.set_experiment(experiment_name)
            exp = self._client.get_experiment_by_name(experiment_name)
            self._client.set_experiment_tag(
                exp.experiment_id, "version_mlflow", mlflow.__version__
            )
            self._client.set_experiment_tag(
                exp.experiment_id, "experiment_created", self._now
            )

    def train(self, run_name):
        """
        Executes the training process, including model fitting, parameter tuning,
        and logging metrics and models in MLflow. Optionally logs plot artifacts.

        :param run_name: The name for the MLflow run.
        :return: A tuple containing the experiment_id and the run_id for the MLflow
                    tracking context.
        """

        with mlflow.start_run(run_name=run_name, nested=True) as run:
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id
            logger.info("MLflow run:")
            logger.info("\trun_id: %s", run_id)
            logger.info("\texperiment_id: %s", experiment_id)
            logger.info(
                "\texperiment_name: %s", self._client.get_experiment(experiment_id).name
            )

            mlflow.set_tag("run_id", run_id)
            mlflow.set_tag("data_path", self.data_path)
            mlflow.set_tag("dataset", "titanic")
            mlflow.set_tag("timestamp", self._now)

            grid = self._pre_process.execute(self._df)
            cv_results = grid.cv_results_
            for params, metric, rank in zip(
                cv_results["params"],
                cv_results["mean_test_score"],
                cv_results["rank_test_score"],
            ):
                with mlflow.start_run(experiment_id=experiment_id, nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metric("accuracy", metric)
                    mlflow.set_tag("rank", rank)

            model = grid.best_estimator_
            predictions = model.predict(self._x_test)

            mlflow.log_params(grid.best_params_)

            accuracy = sklearn.metrics.accuracy_score(self._y_test, predictions)
            precision = sklearn.metrics.precision_score(
                self._y_test, predictions, average="macro"
            )
            recall = sklearn.metrics.recall_score(
                self._y_test, predictions, average="macro"
            )
            f1 = sklearn.metrics.f1_score(self._y_test, predictions, average="macro")

            logger.info("Metrics:")
            logger.info("\taccuracy: %s", accuracy)
            logger.info("\tprecision: %s", precision)
            logger.info("\trecall: %s", recall)
            logger.info("\tf1: %s", f1)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)

            mlflow.sklearn.log_model(model, "model", signature=False)

            if self.log_plot:
                plot_file = "plot.png"
                src.utils.plot.create_confusion_matrix_plot(
                    self._y_test, predictions, plot_file
                )
                mlflow.log_artifact(plot_file)
                if os.path.exists(plot_file):
                    os.remove(plot_file)

        run = self._client.get_run(run_id)
        self._client.set_tag(run_id, "run.info.start_time", run.info.start_time)
        self._client.set_tag(run_id, "run.info.end_time", run.info.end_time)
        self._client.set_tag(
            run_id,
            "run.info._start_time",
            src.utils.timestamp.fmt_ts_millis(run.info.start_time),
        )
        self._client.set_tag(
            run_id,
            "run.info._end_time",
            src.utils.timestamp.fmt_ts_millis(run.info.end_time),
        )

        return experiment_id, run_id


@click.command()
@click.option(
    "--experiment-name",
    help="Experiment name.",
    type=str,
    default="Titanic",
    show_default=True,
)
@click.option("--run-name", help="Run name", type=str, required=False)
@click.option(
    "--data-path", help="Data path.", type=str, default="./data", show_default=True
)
@click.option("--log-plot", help="Log plot", type=bool, default=True, show_default=True)
def main(experiment_name, run_name, data_path, log_plot):
    """
    Command-line interface to set up and run the machine learning experiment.

    :param experiment_name: Name of the MLflow experiment.
    :param run_name: Name of the individual run within the experiment.
    :param data_path: Path to the dataset used for training and testing.
    :param log_plot: Whether to log visualization plots.
    """

    logger.info("Options:")
    for k, v in locals().items():
        logger.info("\t%s: %s", k, v)

    trainer = Trainer(experiment_name, data_path, log_plot)
    experiment_id, run_id = trainer.train(run_name)

    logger.info("\texperiment_id: %s", experiment_id)
    logger.info("\trun_id: %s", run_id)


if __name__ == "__main__":
    main()  # pylint: disable=E1120
