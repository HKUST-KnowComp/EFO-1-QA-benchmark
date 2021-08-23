from .appfoq import (AppFOQEstimator,
                     NLKEstimator,
                     TransEEstimator)
from .foq import FirstOrderQuery, parse_foq_formula
from .estimator_box import BoxEstimator
from .estimator_beta import BetaEstimator, BetaEstimator4V
from .estimator_logic import LogicEstimator
from .foq_v2 import parse_formula, count_depth, count_entities


beta_query = {
    '1p': 'p(e)',
    '2p': 'p(p(e))',
    '3p': 'p(p(p(e)))',
    '2i': 'p(e)&p(e)',
    '3i': 'p(e)&p(e)&p(e)',
    'ip': 'p(p(e)&p(e))',
    'pi': 'p(p(e))&p(e)',
    '2in': 'p(e)-p(e)',
    '3in': 'p(e)&p(e)-p(e)',
    'inp': 'p(p(e)-p(e))',
    'pin': 'p(p(e))-p(e)',
    'pni': 'p(e)-p(p(e))',
    '2u': 'p(e)|p(e)',
    'up': 'p(p(e)|p(e))'
}


beta_query_v2 = {
    '1p': '(p,(e))',
    '2p': '(p,(p,(e)))',
    '3p': '(p,(p,(p,(e))))',
    '2i': '(i,(p,(e)),(p,(e)))',
    '3i': '(i,(p,(e)),(p,(e)),(p,(e)))',
    'ip': '(p,(i,(p,(e)),(p,(e))))',
    'pi': '(i,(p,(p,(e))),(p,(e)))',
    '2in': '(i,(p,(e)),(n,(p,(e))))',
    '3in': '(i,(p,(e)),(p,(e)),(n,(p,(e))))',
    'inp': '(p,(i,(p,(e)),(n,(p,(e)))))',
    'pin': '(i,(p,(p,(e))),(n,(p,(e))))',
    'pni': '(i,(n,(p,(p,(e)))),(p,(e)))',
    '2u': '(u,(p,(e)),(p,(e)))',
    'up': '(p,(u,(p,(e)),(p,(e))))',
    '2u-DNF': '(u,(p,(e)),(p,(e)))',
    'up-DNF': '(u,(p,(p,(e))),(p,(p,(e))))',
    '2u-DM': '(n,(i,(n,(p,(e))),(n,(p,(e)))))',
    'up-DM': '(p,(n,(i,(n,(p,(e))),(n,(p,(e))))))',
    '2D': '(D, (p,(e)), (p,(e)))',
    '3D': '(D, (p,(e)), (p,(e)), (p,(e)))',
    'Dp': '(p, (D, (p,(e)), (p,(e))))',
}
