from .plots import makeplots
from .get import get
from .pair import makeintx
from .word import from_antype

# Get all the data
get()
# Make all the intersections
for spc in ['hcho', 'no2']:
    makeintx(spc)
    for source in ['pandora', 'tropomi', 'airnow']:
        # Make all the plots
        if (source == 'airnow' and spc == 'hcho'):
            continue
        makeplots(source, spc)
        from_antype(source, spc)
