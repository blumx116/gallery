from google.cloud import bigquery

from gallery.constants import DATASET_NAME, PROJECT_NAME, metadata_table, metrics_table

_client = bigquery.Client(project=PROJECT_NAME)


def create_dataset(name: str) -> None:
    dataset = bigquery.Dataset(name)
    dataset.location = "US"
    _client.create_dataset(dataset, exists_ok=True)


if __name__ == "__main__":
    create_dataset(DATASET_NAME)
    _client.create_table(metadata_table, exists_ok=True)
    _client.create_table(metrics_table, exists_ok=True)
