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

        if 'use_flip_channel' not in d['data']:
            d['data']['use_flip_channel'] = False

        if 'apply_clihe' not in d['data']:
            d['data']['apply_clihe'] = False

        if 'flexible_stn' not in d['SATRN']:
            d['SATRN']['flexible_stn'] = {
                'use': False
            }

        if 'use_multi_sample_dropout' not in d['SATRN']:
            d['SATRN']['use_multi_sample_dropout'] = False

        if 'share_transformer' not in d['SATRN']:
            d['SATRN']['share_transformer'] = False
        
        if 'use_separable_cnn' not in d['SATRN']:
            d['SATRN']['use_separable_cnn'] = False

        if 'multi_sample_dropout_ratio' not in d['SATRN']:
            d['SATRN']['multi_sample_dropout_ratio'] = None

        if 'multi_sample_dropout_nums' not in d['SATRN']:
            d['SATRN']['multi_sample_dropout_nums'] = None

        if 'start_dim' not in d['SATRN']['encoder']:
            d['SATRN']['encoder']['start_dim'] = 48

        if 'depth' not in d['SATRN']['encoder']:
            d['SATRN']['encoder']['depth'] = 16

        if 'growth_rate' not in d['SATRN']['encoder']:
            d['SATRN']['encoder']['growth_rate'] = 24

        if 'use_between_ff_layer' not in d['SATRN']['decoder']:
            d['SATRN']['decoder']['use_between_ff_layer'] = False

        if 'emb_dim' not in d['SATRN']['decoder']:
            d['SATRN']['decoder']['emb_dim'] = d['SATRN']['decoder']['hidden_dim']

        if 'DecoderOnly' in d:
            if 'use_256_input' not in d['DecoderOnly']['encoder']:
                d['DecoderOnly']['encoder']['use_256_input'] = False

        if 'set_optimizer_from_checkpoint' not in d['optimizer']:
            d['optimizer']['set_optimizer_from_checkpoint'] = True
            
        self.flags = dict_to_namedtuple(d)
        

    def get(self):
        return self.flags, self.d