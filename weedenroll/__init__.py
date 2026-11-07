"""weedenroll — offline few-shot weed recognition with field enrollment.

Components (map to the paper's system description):
  embed/      Embedding backbone interface + default implementations
  store       On-device prototype store (nearest-prototype classification)
  openset     Distance-based open-set rejection and threshold calibration
  enroll      Field enrollment workflow (k-shot, no gradient updates)
  govern      Local governance layer (confidence, rejection, drift logs)
  bench/      Episodic benchmarks, hardware profiling, cross-platform transfer
  cost        Season-level cost model (edge vs cloud vs retraining)
  report      CIGR-style figures and tables
"""

__version__ = "0.1.0"

from .store import PrototypeStore
from .openset import OpenSetRejector
from .enroll import Enroller
from .govern import GovernanceLogger

__all__ = ["PrototypeStore", "OpenSetRejector", "Enroller", "GovernanceLogger"]
