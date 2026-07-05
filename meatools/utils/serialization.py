#!/usr/bin/env python
"""Shared JSON serialization utilities for meatools."""

import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and scalars."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)
