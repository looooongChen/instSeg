name = "instSeg"

from instSeg.config import Config, ConfigParallel, ConfigCascade
from instSeg.model_cascade import InstSegCascade
from instSeg.model_parallel import InstSegParallel
from instSeg.loader import load_model
from instSeg.stitcher import split, stitch
from instSeg.seg_view import seg_in_tessellation
import instSeg.clustering as clustering
import instSeg.result_analyse as result_analyse


from instSeg.evaluation import Evaluator, Evaluator_Seg
from instSeg import synthesis 
from instSeg import utils 
from instSeg import visualization as vis

from instSeg.pattern_generator import Config as Generator_Config
from instSeg.pattern_generator import Generator as Pattern_Generator

import instSeg.analyzer as analyzer