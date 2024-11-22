from __future__ import annotations
import numpy as np
from pathlib import Path
import pandas as pd
from importlib_resources import files
from enum import Enum
from typing import Callable
from functools import lru_cache, cached_property, cache
from datetime import datetime
from scipy.fft import dct, dst, idct, idst

