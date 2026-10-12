import numpy as np
import json
from pathlib import Path
import datetime


def serialize_value(v):
    # primitives
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v

    # numpy
    if isinstance(v, (np.integer, np.floating)):
        return v.item()

    if isinstance(v, np.ndarray):
        return v.tolist()

    # pathlib
    if isinstance(v, Path):
        return str(v)

    # datetime
    if isinstance(v, (datetime.datetime, datetime.date)):
        return v.isoformat()

    # fallback
    try:
        json.dumps(v)
        return v
    except Exception:
        return str(v)   # last resort
