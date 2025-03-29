__all__ = ['depends']


def depends(outpaths, inpaths, verbose=1):
    """
    Arguments
    ---------
    outpaths : str or list
        List of outpaths may depend on inpaths.
        If str, outpaths = glob.gob(outpaths)
    inpaths : str or list
        List of paths that outpaths may depend on.
        If str, inpaths = glob.gob(inpaths)
    verbose : int
        Level of verbosity

    Returns
    -------
    remake : bool
        True if inpaths are newer than outpaths
    """
    import os
    import glob
    import pandas as pd
    remake = False
    if isinstance(inpaths, str):
        if verbose > 1:
            print(f'INFO::expanded {inpaths}')
        inpaths = sorted(glob.glob(inpaths))
        if verbose > 1:
            print(f'INFO::{inpaths[0]} ... {inpaths[-1]} (n={len(inpaths)})')
    if isinstance(outpaths, str):
        if verbose > 1:
            print(f'INFO::expanded {outpaths}')
        outpaths = sorted(glob.glob(outpaths))
        if verbose > 1:
            print(
                f'INFO::{outpaths[0]} ... {outpaths[-1]} (n={len(outpaths)})'
            )
    if len(outpaths) == 0:
        if verbose > 1:
            print('INFO:: No outpaths found')
        return True
    else:
        inupdated = [
            os.stat(p).st_mtime if os.path.exists(p) else 0 for p in inpaths
        ]
        outupdated = [
            os.stat(p).st_mtime if os.path.exists(p) else 0 for p in outpaths
        ]
        remake = min(outupdated) < max(inupdated)
        if verbose > 0:
            indate = pd.to_datetime(max(inupdated), unit='s')
            outdate = pd.to_datetime(max(outupdated), unit='s')
            print(f'INFO:: Oldest of outpaths (n={len(outpaths)}): {outdate}')
            print(f'INFO:: Newest of inpaths (n={len(inpaths)}): {indate}')
        return remake
