import pyarrow as pa
import numpy as np
import joblib

IN_FILE = "../data/iot2023-8class-http.arrow"
OUT_FILE = "../data/iot2023-8class-http.bin"
OUT_METADATA_FILE = "../data/iot2023-8class-http.metadata.joblib"

meta = {
  "flow_duration": {
    "len": 0,
    "dtype": None
  },
  "burst_tokens": {
    "len": [],
    "dtype": None
  },
  "directions": {
    "len": [],
    "dtype": None
  },
  "bytes": {
    "len": [],
    "dtype": None
  },
  "iats": {
    "len": [],
    "dtype": None
  },
  "counts": {
    "len": [],
    "dtype": None
  },
  "protocol": {
    "len": 0,
    "dtype": None
  },
  "labels": [],
}

with open(OUT_FILE, "wb") as out:
  with pa.OSFile(IN_FILE, "rb") as f:
    with pa.ipc.open_stream(f) as reader:
      table = reader.read_all().combine_chunks()
      

      # Column: flow_duration
      flow_duration = table["flow_duration"].to_numpy()
      meta["flow_duration"]["dtype"] = flow_duration.dtype

      flow_duration_b = flow_duration.tobytes()
      meta["flow_duration"]["len"] = len(flow_duration_b)
      out.write(flow_duration_b)


      # Column: burst_tokens
      burst_tokens = table["burst_tokens"].to_numpy()
      meta["burst_tokens"]["dtype"] = burst_tokens[0][0].dtype

      for i, flow in enumerate(burst_tokens):
        meta["burst_tokens"]["len"].append([])
        for j, burst in enumerate(flow):
          burst_b = burst.tobytes()
          meta["burst_tokens"]["len"][i].append(len(burst_b))
          out.write(burst_b)


      # Column: directions
      directions = table["directions"].to_numpy()
      meta["directions"]["dtype"] = directions[0].dtype

      for i, flow in enumerate(directions):
        flow_b = flow.tobytes()
        meta["directions"]["len"].append(len(flow_b))
        out.write(flow_b)


      # Column: bytes
      bytes_col = table["bytes"].to_numpy()
      meta["bytes"]["dtype"] = bytes_col[0].dtype

      for i, flow in enumerate(bytes_col):
        flow_b = flow.tobytes()
        meta["bytes"]["len"].append(len(flow_b))
        out.write(flow_b)


      # Column: iats
      iats = table["iats"].to_numpy()
      meta["iats"]["dtype"] = iats[0].dtype

      for i, flow in enumerate(iats):
        flow_b = flow.tobytes()
        meta["iats"]["len"].append(len(flow_b))
        out.write(flow_b)


      # Column: counts
      counts = table["counts"].to_numpy()
      meta["counts"]["dtype"] = counts[0].dtype

      for i, flow in enumerate(counts):
        flow_b = flow.tobytes()
        meta["counts"]["len"].append(len(flow_b))
        out.write(flow_b)


      # Column: protocol
      protocol = table["protocol"].to_numpy()
      meta["protocol"]["dtype"] = protocol.dtype

      protocol_b = protocol.tobytes()
      meta["protocol"]["len"] = len(protocol_b)
      out.write(protocol_b)

      # Column: labels
      labels = table["labels"].to_numpy()
      meta["labels"] = labels

joblib.dump(meta, OUT_METADATA_FILE)
