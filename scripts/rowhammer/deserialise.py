import pyarrow as pa
import numpy as np
import joblib

IN_FILE = "../data/5M-iot2023-8class-http.bin"
IN_METADATA_FILE = "../data/iot2023-8class-http.metadata.joblib"
OUT_FILE = "../data/iot2023-8class-http-5m.arrow"

flow_duration_type = pa.uint64()
burst_tokens_type = pa.list_(pa.list_(pa.uint16()))
directions_type = pa.list_(pa.bool_())
bytes_type = pa.list_(pa.uint32())
iats_type = pa.list_(pa.uint64())
counts_type = pa.list_(pa.uint32())
protocol_type = pa.uint16()
label_type = pa.string()

table_schema = pa.schema(
  [
    pa.field("flow_duration", flow_duration_type),
    pa.field("burst_tokens", burst_tokens_type),
    pa.field("directions", directions_type),
    pa.field("bytes", bytes_type),
    pa.field("iats", iats_type),
    pa.field("counts", counts_type),
    pa.field("protocol", protocol_type),
    pa.field("labels", label_type),
  ]
)

meta = joblib.load(IN_METADATA_FILE)

# Reconstruct
with open(IN_FILE, "rb") as infile:

  # Column: flow_duration
  flow_duration = np.frombuffer(
    infile.read(meta["flow_duration"]["len"]),
    dtype=meta["flow_duration"]["dtype"]
  )


  # Column: burst_tokens
  burst_tokens = []
  for i, flow in enumerate(meta["burst_tokens"]["len"]):
    burst_tokens.append([])
    for j, burst_len in enumerate(flow):
      burst_tokens[i].append(np.frombuffer(
        infile.read(burst_len),
        dtype=meta["burst_tokens"]["dtype"]
      ))
  

  # Column: directions
  directions = []
  for i, length in enumerate(meta["directions"]["len"]):
    directions.append(np.frombuffer(
      infile.read(length),
      dtype=meta["directions"]["dtype"]
    ))


  # Column: bytes
  bytes_col = []
  for i, length in enumerate(meta["bytes"]["len"]):
    bytes_col.append(np.frombuffer(
      infile.read(length),
      dtype=meta["bytes"]["dtype"]
    ))


  # Column: iats
  iats = []
  for i, length in enumerate(meta["iats"]["len"]):
    iats.append(np.frombuffer(
      infile.read(length),
      dtype=meta["iats"]["dtype"]
    ))


  # Column: counts
  counts = []
  for i, length in enumerate(meta["counts"]["len"]):
    counts.append(np.frombuffer(
      infile.read(length),
      dtype=meta["counts"]["dtype"]
    ))


  # Column: protocol
  protocol = np.frombuffer(
    infile.read(meta["protocol"]["len"]),
    dtype=meta["protocol"]["dtype"]
  )


  # Column: labels
  labels = meta["labels"]

  new_table = pa.Table.from_pydict(
    {
      "flow_duration": flow_duration,
      "burst_tokens": burst_tokens,
      "directions": directions,
      "bytes": bytes_col,
      "iats": iats,
      "counts": counts,
      "protocol": protocol,
      "labels": labels
    }
  ).cast(table_schema)

with pa.OSFile(OUT_FILE, "wb") as f:
  with pa.ipc.new_stream(f, table_schema) as writer:
    writer.write_table(new_table)