name = "instSeg"

from instSeg.config import Config, ConfigParallel, ConfigCascade
from instSeg.model_cascade import InstSegCascade
from instSeg.model_parallel import InstSegParallel

from instSeg.evaluation import Evaluator
from instSeg import visualization as vis