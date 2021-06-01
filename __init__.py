name = "instSeg"

from instSeg.config import Config, ConfigParallel, ConfigCascade
from instSeg.model_cascade import InstSegCascade
from instSeg.model_parallel import InstSegParallel
from instSeg.loader import load_model
from instSeg.stitcher import split2D, stitch2D
from instSeg.seg_view import seg_in_tessellation

from instSeg.evaluation import Evaluator
from instSeg import visualization as vis