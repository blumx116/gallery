from google.cloud import bigquery


def table_name(table: bigquery.Table) -> str:
    return bigquery.TableReference.from_api_repr(table.reference)
