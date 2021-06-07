"""
Original code from clovaai/SATRN
"""
import os
import yaml
import collections
from copy import deepcopy


def dict_to_namedtuple(d):
    """ Convert dictionary to named tuple.
    """
    FLAGSTuple = collections.namedtuple('FLAGS', sorted(d.keys()) + ['original_config'])

    for k, v in d.items():
        
        if k == 'prefix':
            v = os.path.join('./', v)

        if type(v) is dict:
            d[k] = dict_to_namedtuple(v)

        elif type(v) is str:
            try:
                d[k] = eval(v)
            except:
                d[k] = v

    d['original_config'] = d

    nt = FLAGSTuple(**d)

    return nt


class Flags:
    """ Flags object.
    """

    def __init__(self, config_file):
        try:
            with open(config_file, 'r') as f:
                d = yaml.safe_load(f)
        except:
            d = config_file

        self.d = deepcopy(d)

        if 'curriculum_learning' not in d:
            d['curriculum_learning'] = {
                'using':False,
            }

        if 'is_reverse' not in d['data']:
            d['data']['is_reverse'] = False

        if 'flexible_stn' not in d['SATRN']:
            d['SATRN']['flexible_stn'] = {
                'use': False
            }

        if 'set_optimizer_from_checkpoint' not in d['optimizer']:
            d['optimizer']['set_optimizer_from_checkpoint'] = True
            
        self.flags = dict_to_namedtuple(d)
        

    def get(self):
        return self.flags, self.d