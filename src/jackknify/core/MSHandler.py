import os
import shutil
import numpy as np
from casacore.tables import table as ctab

class MSWrapper:
    """
    Handles interaction with the Measurement Set using casacore.tables.
    """
    def __init__(self, ms_path):
        self.ms_path = ms_path

    def get_data(self, col_name="DATA"):
        """Reads the data column."""
        with ctab(self.ms_path, readonly=True, ack=False) as t:
            if col_name not in t.colnames():
                raise ValueError(f"Column {col_name} not found in {self.ms_path}")
            return t.getcol(col_name)

    def write_column(self, col_name, data, desc_template_col="DATA"):
        """
        Writes data to a specific column. Creates the column if it doesn't exist 
        using the template column description.
        """
        with ctab(self.ms_path, readonly=False, ack=False) as t:
            if col_name in t.colnames():
                t.putcol(col_name, data)
            else:
                desc = t.getcoldesc(desc_template_col)
                desc["name"] = col_name
                desc['comment'] = 'Jackknife_Realization'
                
                # Handle potential tiled shape issues if present in descriptor
                if 'ndim' in desc and desc['ndim'] == -1:
                    pass 

                t.addcols(desc)
                t.putcol(col_name, data)

    def create_copy(self, out_path):
        """Creates a full directory copy of the MS."""
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        shutil.copytree(self.ms_path, out_path)
        return MSWrapper(out_path)
    
    @staticmethod
    def create_test_ms(ms_path, n_rows=100, n_chan=16, n_pol=4):
        """
        Creates a mock Measurement Set table with a DATA column filled with 1s.
        Useful for testing logic without a real MS.
        """
        if os.path.exists(ms_path):
            print(f"Error, {ms_path} already exists. Require empty path destination to create new MS.")
            return
            
        # Define a minimal descriptor sufficient for this tool - opposite to np layout since casa uses fortran ordering
        desc = {
            'DATA': {
                'comment': 'Mock Data',
                'dataManagerGroup': 'StandardStMan',
                'dataManagerType': 'StandardStMan',
                'maxlen': 0,
                'ndim': 2,
                'option': 0,
                'shape': np.array([n_pol, n_chan], dtype=np.int32),
                'valueType': 'complex' # defaults to complex64 (single precision)
            }
        }
        
        # Create the table
        with ctab(ms_path, desc, nrow=n_rows, readonly=False, ack=False) as t:
            # Create data: shape (Rows, Chan, Pol)
            data = np.ones((n_rows, n_chan, n_pol), dtype=np.complex128)
            t.putcol('DATA', data)
            print(f"Created mock MS at {ms_path} with shape {data.shape}")