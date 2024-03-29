name = "instSeg"

from instSeg.config import Config
from instSeg.model import Model
from instSeg.loader import load_model
from instSeg.stitcher import split, stitch
from instSeg.seg_view import seg_in_tessellation
import instSeg.clustering as clustering



from instSeg.evaluation import Evaluator, Evaluator_Seg
from instSeg import synthesis 
from instSeg import utils 
from instSeg import visualization as vis

from instSeg.pattern_generator import Config as Generator_Config
from instSeg.pattern_generator import Generator as Pattern_Generator

import instSeg.result_analyse as result_analyse
import instSeg.model_analyzer as model_analyzer