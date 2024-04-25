__all__ = ['tempo', 'pandora', 'tropomi', 'airnow', 'get_configs', 'server']

import os
import json
from . import tempo
from . import pandora
from . import tropomi
from . import airnow
from getpass import getpass

_configs = {}
pwpath = os.path.expanduser('~/.tempokey')
if os.path.exists(pwpath):
    _tempo_pw = open(pwpath, 'r').read().strip()
else:
    _tempo_pw = getpass('Enter RSIG TEMPO PW:\n')

# If using outside EPA, change server to ofmpub.epa.gov
server = 'maple.hesc.epa.gov'


def get_configs(cfgpath='config.json'):
    if cfgpath not in _configs:
        _configs[cfgpath] = json.load(open(cfgpath, 'r'))
    return _configs[cfgpath]
