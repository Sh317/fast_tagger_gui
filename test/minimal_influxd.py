from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import datetime
import pandas as pd

INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "ZAVlCYaA7Fgop5mH7MpSqbpIXTIdbdJ4_s5R2wNyM3IV5NCT_uKPj2PID-h-a3KDo_9mSo5T65gDG61B8gd8pA=="
INFLUXDB_ORG = "MITBeambox"
INFLUXDB_BUCKET = "DAQ"
NBATCH = 1_000
TOTAL_MAX_POINTS = 5_000_000
MAX_POINTS_FOR_PLOT = 100

REFRESH_RATE = 0.5  # seconds
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

def query_influxdb(minus_time_str, measurement_name):
    query = f"""
    from(bucket: "{INFLUXDB_BUCKET}")
    |> range(start: {minus_time_str})
    |> filter(fn: (r) => r._measurement == "tagger")
    |> filter(fn: (r) => r.type == "{measurement_name}")
    |> tail(n: {NBATCH})
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    |> keep(columns: ["_time", "bunch", "n_events", "channel", "time_offset", "id_timestamp", "wn_1", "wn_2", "wn_3", "wn_4", "voltage", "spectr_peak", "trigger_rate", "event_rate"])
    """
    try:
        result = client.query_api().query(query=query, org=INFLUXDB_ORG)
        records = []
        for table in result:
            for record in table.records:
                records.append(record.values)
        df = pd.DataFrame(records)#.dropna(how="all")
        return df
    except Exception as e:
        print(f"Error querying InfluxDB: {e}")
        return pd.DataFrame(columns=[
            "_time", "bunch", "n_events", "channel", "time_offset", "id_timestamp",
            "wn_1", "wn_2", "wn_3", "wn_4", "voltage", "spectr_peak", "trigger_rate", "event_rate"
        ])

if __name__ == "__main__":
    minus_time_str = datetime.strptime(measurement_name, "%Y_%m_%d_%H_%M_%S").strftime("%Y-%m-%dT%H:%M:%SZ")
    new_data = query_influxdb(minus_time_str, measurement_name)
    print(new_data)