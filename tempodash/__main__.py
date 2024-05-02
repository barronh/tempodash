import gc
from .cfg import libc
from .plots import makeplots
from .get import get
from .pair import makeintx
from .word import from_antype

# Get all the data
get()
# Make all the intersections
for spc in ['no2', 'hcho']:
    makeintx(spc)
    for source in ['pandora', 'tropomi_offl', 'airnow']:
        # Make all the plots
        if (source == 'airnow' and spc == 'hcho'):
            continue
        try:
            makeplots(source, spc)
            from_antype(source, spc)
        except Exception as e:
            print(source, spc, str(e))
        gc.collect()
        libc.malloc_trim(1)
