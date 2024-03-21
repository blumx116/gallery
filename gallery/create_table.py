from google.cloud import bigquery

from gallery.constants import DATASET_NAME, PROJECT_NAME, TEST_TABLE

client = bigquery.Client(project=PROJECT_NAME)

dataset = bigquery.Dataset(f"{PROJECT_NAME}.{DATASET_NAME}")
dataset.location = "US"

client.create_dataset(dataset, exists_ok=True)

schema = [
    bigquery.SchemaField("epoch", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("samples", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("step", "INTEGER", mode="REQUIRED"),
    batches_per_sec
    throughput/batches_per_sec	
    throughput/device/batches_per_sec	
    throughput/device/flops_per_sec	
    throughput/device/mfu	
    throughput/device/samples_per_sec	
    throughput/device/tokens_per_sec	
    throughput/flops_per_sec	
    throughput/samples_per_sec	
    throughput/tokens_per_sec	
    time/total	
    time/train	
    time/val	
    train_loss
]

table = bigquery.Table(f"{PROJECT_NAME}.{DATASET_NAME}.{TEST_TABLE}", schema)
client.create_table(table, exists_ok=True)


