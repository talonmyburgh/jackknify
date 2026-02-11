import os
from astropy.io import fits

class FitsWrapper:
    """
    Handles interaction with FITS files using astropy.
    """
    def __init__(self, filepath):
        self.filepath = filepath

    @property
    def data(self):
        """Returns the data array from the primary HDU."""
        return fits.getdata(self.filepath)

    @property
    def header(self):
        """Returns the header from the primary HDU."""
        return fits.getheader(self.filepath)

    @staticmethod
    def write_cube(out_path, data, header, overwrite=True):
        """
        Writes a data array and header to a FITS file.
        """
        hdu = fits.PrimaryHDU(data=data, header=header)
        hdu.writeto(out_path, overwrite=overwrite)