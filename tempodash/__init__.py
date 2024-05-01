__all__ = [
    'tempo', 'pandora', 'tropomi', 'airnow', 'word', 'server',
    'util'
]

from . import util
from . import tempo
from . import pandora
from . import tropomi
from . import airnow
from . import word

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# If using outside EPA, change server to ofmpub.epa.gov
# server = 'maple.hesc.epa.gov'
server = 'ofmpub.epa.gov'
