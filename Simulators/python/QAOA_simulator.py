from __future__ import annotations
#Allows you to reference classes and functions that are defined later in the code.
# which means that type hints can be written as strings and will be evaluated only when needed
from collections.abc import Sequence
import numpy as np
from Base import QAOAFastSimulatorBase, CostsType, TermsType, ParamType
from ...precoputation.numpy_vectorized import precompute_vectorized_cpu_parallel
from .QAOA_simulator import apply_qaoa_furx

