"""
JSON serialization utilities for trading data structures.
"""

import json
import pandas as pd


def json_serializable(obj):
    """Convert objects to JSON serializable format."""
    if obj is None:
        return None
    elif isinstance(obj, (dict, list, str, int, float, bool)):
        if isinstance(obj, dict):
            return {k: json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [json_serializable(item) for item in obj]
        else:
            return obj
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif hasattr(obj, 'isoformat'):
        return obj.isoformat()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    else:
        try:
            if hasattr(obj, 'to_dict') and not isinstance(obj, dict):
                return json_serializable(obj.to_dict())
            elif hasattr(obj, '__dict__'):
                return json_serializable(obj.__dict__)
            else:
                return str(obj)
        except Exception:
            return str(obj)


def safe_json_dumps(obj):
    """Safely convert object to JSON string with custom serialization."""
    def default_serializer(o):
        return json_serializable(o)

    return json.dumps(obj, default=default_serializer, ensure_ascii=False, indent=2)
