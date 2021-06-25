
"""
Created on Friday June 25, 2021

@author: Hamid-mojtabavi

Temperature index mass balance with debri covers on the Huss flowlines
"""

# Built ins
import logging
# External libs
import numpy as np
import rasterio
import os
import xarray as xr
import pandas as pd
import netCDF4
from scipy.interpolate import interp1d
from scipy import optimize as optimization
# Locals
import oggm
import oggm.cfg as cfg
from oggm.cfg import SEC_IN_YEAR, SEC_IN_MONTH
from oggm.utils import (SuperclassMeta, lazy_property, floatyear_to_date,
                        date_to_floatyear, monthly_timeseries, ncDataset,
                        tolist, clip_min, clip_max, clip_array)
from oggm import utils                       
from oggm.exceptions import InvalidWorkflowError, InvalidParamsError
from oggm import entity_task
from oggm.core.gis import rasterio_to_gdir
from oggm.utils import ncDataset

# Module logger
log = logging.getLogger(__name__)


# Add debris data to their Glacier directories, same as in PyGEM (David Rounce), https://github.com/drounce/PyGEM/blob/master/pygem/shop/debris.py
# Debris data can download from https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020GL091311
#debris_fp = 'https://cluster.klima.uni-bremen.de/~smojtabavi/DATA/hd_glob/'
debris_fp = '/home/users/smojtabavi/DATA/hd_glob/' # this needs to be changed if working on another computer


# Add the new name "hd" to the list of things that the GlacierDirectory understands
if not 'debris_hd' in cfg.BASENAMES:
    cfg.BASENAMES['debris_hd'] = ('debris_hd.tif', 'Raster of debris thickness data')
if not 'debris_ed' in cfg.BASENAMES:
    cfg.BASENAMES['debris_ed'] = ('debris_ed.tif', 'Raster of debris enhancement factor data')

@entity_task(log, writes=['debris_hd', 'debris_ed'])
def debris_to_gdir(gdir, debris_dir=debris_fp, add_to_gridded=True, hd_max=5, hd_min=0, ed_max=10, ed_min=0):
    """Reproject the debris thickness and enhancement factor files to the given glacier directory
    
    Variables are exported as new files in the glacier directory.
    Reprojecting debris data from one map proj to another is done. 
    We use bilinear interpolation to reproject the velocities to the local glacier map.
    
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """

    assert os.path.exists(debris_dir), "Error: debris directory does not exist."

    hd_dir = debris_dir + 'hd_tifs/' + gdir.rgi_region + '/'
    ed_dir = debris_dir + 'Ed_tifs/' + gdir.rgi_region + '/'
    
    glac_str_nolead = str(int(gdir.rgi_region)) + '.' + gdir.rgi_id.split('-')[1].split('.')[1]
    
    # If debris thickness data exists, then write to glacier directory
    if os.path.exists(hd_dir + glac_str_nolead + '_hdts_m.tif'):
        hd_fn = hd_dir + glac_str_nolead + '_hdts_m.tif'
    elif os.path.exists(hd_dir + glac_str_nolead + '_hdts_m_extrap.tif'):
        hd_fn = hd_dir + glac_str_nolead + '_hdts_m_extrap.tif'
    else: 
        hd_fn = None
        
    if hd_fn is not None:
        rasterio_to_gdir(gdir, hd_fn, 'debris_hd', resampling='bilinear')
    if add_to_gridded and hd_fn is not None:
        output_fn = gdir.get_filepath('debris_hd')
        
        # append the debris data to the gridded dataset
        with rasterio.open(output_fn) as src:
            grids_file = gdir.get_filepath('gridded_data')
            with ncDataset(grids_file, 'a') as nc:
                # Mask values
                glacier_mask = nc['glacier_mask'][:] 
                data = src.read(1) * glacier_mask
                data[data>hd_max] = 0
                data[data<hd_min] = 0
                
                # Write data
                vn = 'debris_hd'
                if vn in nc.variables:
                    v = nc.variables[vn]
                else:
                    v = nc.createVariable(vn, 'f8', ('y', 'x', ), zlib=True)
                v.units = 'm'
                v.long_name = 'Debris thicknness'
                v[:] = data
        
    # If debris enhancement factor data exists, then write to glacier directory
    if os.path.exists(ed_dir + glac_str_nolead + '_meltfactor.tif'):
        ed_fn = ed_dir + glac_str_nolead + '_meltfactor.tif'
    elif os.path.exists(ed_dir + glac_str_nolead + '_meltfactor_extrap.tif'):
        ed_fn = ed_dir + glac_str_nolead + '_meltfactor_extrap.tif'
    else: 
        ed_fn = None
        
    if ed_fn is not None:
        rasterio_to_gdir(gdir, ed_fn, 'debris_ed', resampling='bilinear')
    if add_to_gridded and ed_fn is not None:
        output_fn = gdir.get_filepath('debris_ed')
        # append the debris data to the gridded dataset
        with rasterio.open(output_fn) as src:
            grids_file = gdir.get_filepath('gridded_data')
            with ncDataset(grids_file, 'a') as nc:
                # Mask values
                glacier_mask = nc['glacier_mask'][:] 
                data = src.read(1) * glacier_mask
                data[data>ed_max] = 1
                data[data<ed_min] = 1
                # Write data
                vn = 'debris_ed'
                if vn in nc.variables:
                    v = nc.variables[vn]
                else:
                    v = nc.createVariable(vn, 'f8', ('y', 'x', ), zlib=True)
                v.units = '-'
                v.long_name = 'Debris enhancement factor'
                v[:] = data

# Here, I have combined two tasks from OGGM centerline "elevation_band_flowline" and "fixed_dx_elevation_band_flowline"
from oggm.core.centerlines import Centerline
@entity_task(log, writes=['elevation_band_flowline', 'inversion_flowlines'])
def elevation_band_flowline_debris(gdir, bin_variables=None, preserve_totals=True):
    """Compute "squeezed" or "collapsed" glacier flowlines from Huss 2012 
    and, Converts the "collapsed" flowline into a regular "inversion flowline

    This writes out a table of along glacier bins, strictly following the
    method described in Werder, M. A., Huss, M., Paul, F., Dehecq, A. and
    Farinotti, D.: A Bayesian ice thickness estimation model for large-scale
    applications, J. Glaciol., 1–16, doi:10.1017/jog.2019.93, 2019.

    The only parameter is cfg.PARAMS['elevation_band_flowline_binsize'],
    which is 30m in Werder et al and 10m in Huss&Farinotti2012.

    Currently the bands are assumed to have a rectangular bed.

    Before calling this task you should run `tasks.define_glacier_region`
    and `gis.simple_glacier_masks`. It then interpolates onto a regular 
    grid with the same dx as the one that OGGM would choose
    (cfg.PARAMS['flowline_dx'] * map_dx).

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    bin_variables : str or list of str
        variables (debris_ed or debris_hd) to add to the binned flowline (csv file: gdir.get_filepath('elevation_band_flowline').)
        variables (debris_ed or debris_hd) to add to the interpolated flowline (will be stored in a new
        csv file: gdir.get_filepath('elevation_band_flowline',
        filesuffix='_fixed_dx').
    preserve_totals : bool or list of bool
        wether or not to preserve the variables totals (e.g. volume)
    """

    # Variables (debris_ed or debris_hd)
    bin_variables = [] if bin_variables is None else utils.tolist(bin_variables)
    out_vars = []
    out_totals = []
    grids_file = gdir.get_filepath('gridded_data')
    with utils.ncDataset(grids_file) as nc:
        glacier_mask = nc.variables['glacier_mask'][:] == 1
        topo = nc.variables['topo_smoothed'][:]

        # Check if there and do not raise when not available
        keep = []
        for var in bin_variables:
            if var in nc.variables:
                keep.append(var)
            else:
                log.warning('{}: var `{}` not found in gridded_data.'
                            ''.format(gdir.rgi_id, var))
        bin_variables = keep
        for var in bin_variables:
            data = nc.variables[var][:]
            out_totals.append(np.nansum(data) * gdir.grid.dx ** 2)
            out_vars.append(data[glacier_mask])

    preserve_totals = utils.tolist(preserve_totals, length=len(bin_variables))

    # Slope
    sy, sx = np.gradient(topo, gdir.grid.dx)
    slope = np.arctan(np.sqrt(sx ** 2 + sy ** 2))

    # Clip following Werder et al 2019
    slope = utils.clip_array(slope, np.deg2rad(0.4), np.deg2rad(60))

    topo = topo[glacier_mask]
    slope = slope[glacier_mask]

    bsize = cfg.PARAMS['elevation_band_flowline_binsize']

    # Make nice bins ensuring to cover the full range with the given bin size
    maxb = utils.nicenumber(np.max(topo), bsize)
    minb = utils.nicenumber(np.min(topo), bsize, lower=True)
    bins = np.arange(minb, maxb + 0.01, bsize)

    # Some useful constants
    min_alpha = np.deg2rad(0.4)
    max_alpha = np.deg2rad(60)

    if len(bins) < 3:
        # Very low elevation range
        bsize = cfg.PARAMS['elevation_band_flowline_binsize'] / 3
        maxb = utils.nicenumber(np.max(topo), bsize)
        minb = utils.nicenumber(np.min(topo), bsize, lower=True)
        bins = np.arange(minb, maxb + 0.01, bsize)
        if len(bins) < 3:
            # Ok this just not gonna work
            raise InvalidDEMError('({}) DEM altidude range too small.'
                                  .format(gdir.rgi_id))

    # Go - binning
    df = pd.DataFrame()
    topo_digi = np.digitize(topo, bins) - 1  # I prefer the left
    for bi in range(len(bins) - 1):
        # The coordinates of the current bin
        bin_coords = topo_digi == bi

        # Bin area
        bin_area = np.sum(bin_coords) * gdir.grid.dx ** 2
        if bin_area == 0:
            # Ignored in this case - which I believe is strange because deltaH
            # should be larger for the previous bin, but this is what they do
            # according to Zekollari 2019 review
            df.loc[bi, 'area'] = np.NaN
            continue
        df.loc[bi, 'area'] = bin_area

        # Bin average elevation
        df.loc[bi, 'mean_elevation'] = np.mean(topo[bin_coords])

        # Bin averge slope
        # there are a few more shenanigans here described in Werder et al 2019
        s_bin = slope[bin_coords]
        # between the 5% percentile and the x% percentile where x is some magic
        qmin = np.quantile(s_bin, 0.05)
        x = max(2 * np.quantile(s_bin, 0.2) / np.quantile(s_bin, 0.8), 0.55)
        x = min(x, 0.95)
        qmax = np.quantile(s_bin, x)
        sel_s_bin = s_bin[(s_bin >= qmin) & (s_bin <= qmax)]
        if len(sel_s_bin) == 0:
            # This can happen when n pix is small. In this case we just avg
            avg_s = np.mean(s_bin)
        else:
            avg_s = np.mean(sel_s_bin)

        # Final clip as in Werder et al 2019
        df.loc[bi, 'slope'] = utils.clip_scalar(avg_s, min_alpha, max_alpha)

        # Binned variables
        for var, data in zip(bin_variables, out_vars):
            df.loc[bi, var] = np.nanmean(data[bin_coords])

    # The grid point's grid spacing and widths
    df['bin_elevation'] = (bins[1:] + bins[:-1]) / 2
    df['dx'] = bsize / np.tan(df['slope'])
    df['width'] = df['area'] / df['dx']

    # Remove possible NaNs from above
    df = df.dropna()

    # Check for binned vars
    for var, data, in_total, do_p in zip(bin_variables, out_vars, out_totals,
                                         preserve_totals):
        if do_p:
            out_total = np.nansum(df[var] * df['area'])
            if out_total > 0:
                df[var] *= in_total / out_total

    # In OGGM we go from top to bottom
    df = df[::-1]

    # The x coordinate in meter - this is a bit arbitrary but we put it at the
    # center of the irregular grid (better for interpolation later
    dx = df['dx'].values
    dx_points = np.append(dx[0]/2, (dx[:-1] + dx[1:]) / 2)
    df.index = np.cumsum(dx_points)
    df.index.name = 'dis_along_flowline'

    # Store and return
    df.to_csv(gdir.get_filepath('elevation_band_flowline'))

    # From here "fixed_dx_elevation_band_flowline" task

    df = pd.read_csv(gdir.get_filepath('elevation_band_flowline'), index_col=0)

    map_dx = gdir.grid.dx
    dx = cfg.PARAMS['flowline_dx']
    dx_meter = dx * map_dx
    nx = int(df.dx.sum() / dx_meter)
    dis_along_flowline = dx_meter / 2 + np.arange(nx) * dx_meter

    while dis_along_flowline[-1] > df.index[-1]:
        # do not extrapolate
        dis_along_flowline = dis_along_flowline[:-1]

    while dis_along_flowline[0] < df.index[0]:
        # do not extrapolate
        dis_along_flowline = dis_along_flowline[1:]

    nx = len(dis_along_flowline)

    # Interpolate the data we need
    hgts = np.interp(dis_along_flowline, df.index, df['mean_elevation'])
    widths_m = np.interp(dis_along_flowline, df.index, df['width'])

    # Correct the widths - area preserving
    area = np.sum(widths_m * dx_meter)
    fac = gdir.rgi_area_m2 / area
    log.debug('(%s) corrected widths with a factor %.2f', gdir.rgi_id, fac)
    widths_m *= fac

    # Additional vars
    if bin_variables is not None:
        bin_variables = utils.tolist(bin_variables)

        # Check if there and do not raise when not available
        keep = []
        for var in bin_variables:
            if var in df:
                keep.append(var)
            else:
                log.warning('{}: var `{}` not found in gridded_data.'
                            ''.format(gdir.rgi_id, var))
        bin_variables = keep

        preserve_totals = utils.tolist(preserve_totals,
                                       length=len(bin_variables))
        odf = pd.DataFrame(index=dis_along_flowline)
        odf.index.name = 'dis_along_flowline'
        odf['widths_m'] = widths_m
        odf['area_m2'] = widths_m * dx_meter
        for var, do_p in zip(bin_variables, preserve_totals):
            interp = np.interp(dis_along_flowline, df.index, df[var])
            if do_p:
                in_total = np.nansum(df[var] * df['area'])
                out_total = np.nansum(interp * widths_m * dx_meter)
                if out_total > 0:
                    interp *= in_total / out_total
            odf[var] = interp
        odf.to_csv(gdir.get_filepath('elevation_band_flowline',
                                     filesuffix='_fixed_dx'))

    # Write as a Centerline object
    fl = Centerline(None, dx=dx, surface_h=hgts, rgi_id=gdir.rgi_id,
                    map_dx=map_dx)
    fl.order = 0
    fl.widths = widths_m / map_dx
    fl.is_rectangular = np.zeros(nx, dtype=bool)
    fl.is_trapezoid = np.ones(nx, dtype=bool)

    if gdir.is_tidewater:
        fl.is_rectangular[-5:] = True
        fl.is_trapezoid[-5:] = False

    gdir.write_pickle([fl], 'inversion_flowlines')
    gdir.add_to_diagnostics('flowline_type', 'elevation_band')


########### Mass balance models
class MassBalanceModel(object, metaclass=SuperclassMeta):
    """Interface and common logic for all mass balance models used in OGGM.

    All mass-balance models should implement this interface.

    Attributes
    ----------
    valid_bounds : [float, float]
        The altitudinal bounds where the MassBalanceModel is valid. This is
        necessary for automated ELA search.
    """

    def __init__(self):
        """ Initialize."""
        self.valid_bounds = None
        self.hemisphere = None
        self.rho = cfg.PARAMS['ice_density']

    # TODO: remove this in OGGM v1.5
    @property
    def prcp_bias(self):
        raise AttributeError('prcp_bias has been renamed to prcp_fac as it is '
                             'a multiplicative factor, please use prcp_fac '
                             'instead.')

    @prcp_bias.setter
    def prcp_bias(self, new_prcp_fac):
        raise AttributeError('prcp_bias has been renamed to prcp_fac as it is '
                             'a multiplicative factor. If you want to '
                             'change the precipitation scaling factor use '
                             'prcp_fac instead.')

    def get_monthly_mb(self, heights, year=None, fl_id=None, fls=None):
        """Monthly mass-balance at given altitude(s) for a moment in time.

        Units: [m s-1], or meters of ice per second

        Note: `year` is optional because some simpler models have no time
        component.

        Parameters
        ----------
        heights: ndarray
            the atitudes at which the mass-balance will be computed
        year: float, optional
            the time (in the "hydrological floating year" convention)
        fl_id: float, optional
            the index of the flowline in the fls array (might be ignored
            by some MB models)
        fls: list of flowline instances, optional
            the flowlines array, in case the MB model implementation needs
            to know details about the glacier geometry at the moment the
            MB model is called

        Returns
        -------
        the mass-balance (same dim as `heights`) (units: [m s-1])
        """
        raise NotImplementedError()

    def get_annual_mb(self, heights, year=None, fl_id=None, fls=None):
        """Like `self.get_monthly_mb()`, but for annual MB.

        For some simpler mass-balance models ``get_monthly_mb()` and
        `get_annual_mb()`` can be equivalent.

        Units: [m s-1], or meters of ice per second

        Note: `year` is optional because some simpler models have no time
        component.

        Parameters
        ----------
        heights: ndarray
            the altitudes at which the mass-balance will be computed
        year: float, optional
            the time (in the "floating year" convention)
        fl_id: float, optional
            the index of the flowline in the fls array (might be ignored
            by some MB models)
        fls: list of flowline instances, optional
            the flowlines array, in case the MB model implementation needs
            to know details about the glacier geometry at the moment the
            MB model is called

        Returns
        -------
        the mass-balance (same dim as `heights`) (units: [m s-1])
        """
        raise NotImplementedError()

    def get_annual_mb_debris(self, heights, year=None, fl_id=None, fls=None):
        """Like `self.get_monthly_mb()`, but for annual MB with debris.

        For some simpler mass-balance models ``get_monthly_mb()` and
        `get_annual_mb()`` can be equivalent.

        Units: [m s-1], or meters of ice per second

        Note: `year` is optional because some simpler models have no time
        component.

        Parameters
        ----------
        heights: ndarray
            the altitudes at which the mass-balance will be computed
        year: float, optional
            the time (in the "floating year" convention)
        fl_id: float, optional
            the index of the flowline in the fls array (might be ignored
            by some MB models)
        fls: list of flowline instances, optional
            the flowlines array, in case the MB model implementation needs
            to know details about the glacier geometry at the moment the
            MB model is called

        Returns
        -------
        the mass-balance (same dim as `heights`) (units: [m s-1])
        """
        
        raise NotImplementedError()

    def get_specific_mb(self, heights=None, widths=None, fls=None,
                        year=None):
        """Specific mb for this year and a specific glacier geometry.

         Units: [mm w.e. yr-1], or millimeter water equivalent per year

        Parameters
        ----------
        heights: ndarray
            the altitudes at which the mass-balance will be computed.
            Overridden by ``fls`` if provided
        widths: ndarray
            the widths of the flowline (necessary for the weighted average).
            Overridden by ``fls`` if provided
        fls: list of flowline instances, optional
            Another way to get heights and widths - overrides them if
            provided.
        year: float, optional
            the time (in the "hydrological floating year" convention)

        Returns
        -------
        the specific mass-balance (units: mm w.e. yr-1)
        """

        if len(np.atleast_1d(year)) > 1:
            out = [self.get_specific_mb(heights=heights, widths=widths,
                                        fls=fls, year=yr)
                   for yr in year]
            return np.asarray(out)

        if fls is not None:
            mbs = []
            widths = []
            for i, fl in enumerate(fls):
                _widths = fl.widths
                try:
                    # For rect and parabola don't compute spec mb
                    _widths = np.where(fl.thick > 0, _widths, 0)
                except AttributeError:
                    pass
                widths = np.append(widths, _widths)
                mbs = np.append(mbs, self.get_annual_mb(fl.surface_h,
                                                        fls=fls, fl_id=i,
                                                        year=year))
        else:
            mbs = self.get_annual_mb(heights, year=year)

        return np.average(mbs, weights=widths) * SEC_IN_YEAR * self.rho

    def get_specific_mb_debris(self, heights=None, widths=None, fls=None,
                        year=None):
        """Specific mb for this year and a specific glacier geometry with debris.

         Units: [mm w.e. yr-1], or millimeter water equivalent per year

        Parameters
        ----------
        heights: ndarray
            the altitudes at which the mass-balance will be computed.
            Overridden by ``fls`` if provided
        widths: ndarray
            the widths of the flowline (necessary for the weighted average).
            Overridden by ``fls`` if provided
        fls: list of flowline instances, optional
            Another way to get heights and widths - overrides them if
            provided.
        year: float, optional
            the time (in the "hydrological floating year" convention)

        Returns
        -------
        the specific mass-balance (units: mm w.e. yr-1)
        """

        if len(np.atleast_1d(year)) > 1:
            out = [self.get_specific_mb_debris(heights=heights, widths=widths,
                                        fls=fls, year=yr)
                   for yr in year]
            return np.asarray(out)

        if fls is not None:
            mbs = []
            widths = []
            for i, fl in enumerate(fls):
                _widths = fl.widths
                try:
                    # For rect and parabola don't compute spec mb
                    _widths = np.where(fl.thick > 0, _widths, 0)
                except AttributeError:
                    pass
                widths = np.append(widths, _widths)
                mbs = np.append(mbs, self.get_annual_mb_debris(fl.surface_h,
                                                        fls=fls, fl_id=i,
                                                        year=year))
        else:
            mbs = self.get_annual_mb_debris(heights, year=year)

        return np.average(mbs, weights=widths) * SEC_IN_YEAR * self.rho

    

    def get_ela(self, year=None, **kwargs):
        """Compute the equilibrium line altitude for this year

        Parameters
        ----------
        year: float, optional
            the time (in the "hydrological floating year" convention)
        **kwargs: any other keyword argument accepted by self.get_annual_mb
        Returns
        -------
        the equilibrium line altitude (ELA, units: m)
        """

        if len(np.atleast_1d(year)) > 1:
            return np.asarray([self.get_ela(year=yr, **kwargs) for yr in year])

        if self.valid_bounds is None:
            raise ValueError('attribute `valid_bounds` needs to be '
                             'set for the ELA computation.')

        # Check for invalid ELAs
        b0, b1 = self.valid_bounds
        if (np.any(~np.isfinite(
                self.get_annual_mb([b0, b1], year=year, **kwargs))) or
                (self.get_annual_mb([b0], year=year, **kwargs)[0] > 0) or
                (self.get_annual_mb([b1], year=year, **kwargs)[0] < 0)):
            return np.NaN

        def to_minimize(x):
            return (self.get_annual_mb([x], year=year, **kwargs)[0] *
                    SEC_IN_YEAR * self.rho)
        return optimization.brentq(to_minimize, *self.valid_bounds, xtol=0.1)


class PastMassBalance(MassBalanceModel):
    """Mass balance during the climate data period."""

    def __init__(self, gdir, fls=None, mu_star=None, bias=None,
                 filename='climate_historical', input_filesuffix='',
                 repeat=False, ys=None, ye=None, check_calib_params=True, use_inversion_flowlines=True, **kwargs):
        """Initialize.
        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        mu_star : float, optional
            set to the alternative value of mu* you want to use
            (the default is to use the calibrated value).
        bias : float, optional
            set to the alternative value of the calibration bias [mm we yr-1]
            you want to use (the default is to use the calibrated value)
            Note that this bias is *substracted* from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        filename : str, optional
            set to a different BASENAME if you want to use alternative climate
            data.
        input_filesuffix : str
            the file suffix of the input climate file
        repeat : bool
            Whether the climate period given by [ys, ye] should be repeated
            indefinitely in a circular way
        ys : int
            The start of the climate period where the MB model is valid
            (default: the period with available data)
        ye : int
            The end of the climate period where the MB model is valid
            (default: the period with available data)
        check_calib_params : bool
            OGGM will try hard not to use wrongly calibrated mu* by checking
            the parameters used during calibration and the ones you are
            using at run time. If they don't match, it will raise an error.
            Set to False to suppress this check.

        Attributes
        ----------
        temp_bias : float, default 0
            Add a temperature bias to the time series
        prcp_fac : float, default cfg.PARAMS['prcp_scaling_factor']
            Precipitation factor to the time series (called factor to make clear
             that it is a multiplicative factor in contrast to the additive
             `temp_bias`)
        """


        # Read in the flowlines
        if use_inversion_flowlines:
            fls = gdir.read_pickle('inversion_flowlines')

        if fls is None:
            try:
                fls = gdir.read_pickle('model_flowlines')
            except FileNotFoundError:
                raise InvalidWorkflowError('Need a valid `model_flowlines` '
                                           'file. If you explicitly want to '
                                           'use `inversion_flowlines`, set '
                                           'use_inversion_flowlines=True.')

        self.fls = fls
        _y0 = kwargs.get('y0', None)

        # Read debris CSV files

        if not 'elevation_band_flowline_fixed_dx' in cfg.BASENAMES:
            cfg.BASENAMES['elevation_band_flowline_fixed_dx'] = ('elevation_band_flowline_fixed_dx.csv', 'enhanments factor data')
        
        fls_d = pd.read_csv(gdir.get_filepath('elevation_band_flowline_fixed_dx'), index_col=0)
        
        if fls_d  is None:
            try:
                fls_d = pd.read_csv(gdir.get_filepath('elevation_band_flowline_fixed_dx'), index_col=0)
            except FileNotFoundError:
                raise InvalidWorkflowError('Need a valid `elevation_band_flowline_fixed_dx.csv` ')

        self.fls_d = fls_d

        super(PastMassBalance, self).__init__()
        self.valid_bounds = [-1e4, 2e4]  # in m
        if mu_star is None:
            df = gdir.read_json('local_mustar')
            mu_star = df['mu_star_glacierwide']
            if check_calib_params:
                if not df['mu_star_allsame']:
                    msg = ('You seem to use the glacier-wide mu* to compute '
                           'the mass-balance although this glacier has '
                           'different mu* for its flowlines. Set '
                           '`check_calib_params=False` to prevent this '
                           'error.')
                    raise InvalidWorkflowError(msg)

        if bias is None:
            if cfg.PARAMS['use_bias_for_run']:
                df = gdir.read_json('local_mustar')
                bias = df['bias']
            else:
                bias = 0.

        self.mu_star = mu_star
        self.bias = bias

        # Parameters
        self.t_solid = cfg.PARAMS['temp_all_solid']
        self.t_liq = cfg.PARAMS['temp_all_liq']
        self.t_melt = cfg.PARAMS['temp_melt']
        prcp_fac = cfg.PARAMS['prcp_scaling_factor']
        # check if valid prcp_fac is used
        if prcp_fac <= 0:
            raise InvalidParamsError('prcp_fac has to be above zero!')
        default_grad = cfg.PARAMS['temp_default_gradient']

        # Check the climate related params to the GlacierDir to make sure
        if check_calib_params:
            mb_calib = gdir.get_climate_info()['mb_calib_params']
            for k, v in mb_calib.items():
                if v != cfg.PARAMS[k]:
                    msg = ('You seem to use different mass-balance parameters '
                           'than used for the calibration. Set '
                           '`check_calib_params=False` to ignore this '
                           'warning.')
                    raise InvalidWorkflowError(msg)

        # Public attrs
        self.hemisphere = gdir.hemisphere
        self.repeat = repeat

        # Private attrs
        # to allow prcp_fac to be changed after instantiation
        # prescribe the prcp_fac as it is instantiated
        self._prcp_fac = prcp_fac
        # same for temp bias
        self._temp_bias = 0.

        # Read file
        fpath = gdir.get_filepath(filename, filesuffix=input_filesuffix)
        with ncDataset(fpath, mode='r') as nc:
            # time
            time = nc.variables['time']
            time = netCDF4.num2date(time[:], time.units)
            ny, r = divmod(len(time), 12)
            if r != 0:
                raise ValueError('Climate data should be N full years')
            # This is where we switch to hydro float year format
            # Last year gives the tone of the hydro year
            self.years = np.repeat(np.arange(time[-1].year-ny+1,
                                             time[-1].year+1), 12)
            self.months = np.tile(np.arange(1, 13), ny)
            # Read timeseries and correct it
            self.temp = nc.variables['temp'][:].astype(np.float64) + self._temp_bias
            self.prcp = nc.variables['prcp'][:].astype(np.float64) * self._prcp_fac
            if 'gradient' in nc.variables:
                grad = nc.variables['gradient'][:].astype(np.float64)
                # Security for stuff that can happen with local gradients
                g_minmax = cfg.PARAMS['temp_local_gradient_bounds']
                grad = np.where(~np.isfinite(grad), default_grad, grad)
                grad = clip_array(grad, g_minmax[0], g_minmax[1])
            else:
                grad = self.prcp * 0 + default_grad
            self.grad = grad
            self.ref_hgt = nc.ref_hgt
            self.ys = self.years[0] if ys is None else ys
            self.ye = self.years[-1] if ye is None else ye

    # adds the possibility of changing prcp_fac
    # after instantiation with properly changing the prcp time series
    @property
    def prcp_fac(self):
        return self._prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, new_prcp_fac):
        # just to check that no invalid prcp_factors are used
        if new_prcp_fac <= 0:
            raise InvalidParamsError('prcp_fac has to be above zero!')
        self.prcp *= new_prcp_fac / self._prcp_fac
        # update old prcp_fac in order that it can be updated again ...
        self._prcp_fac = new_prcp_fac

    # same for temp_bias:
    @property
    def temp_bias(self):
        return self._temp_bias

    @temp_bias.setter
    def temp_bias(self, new_temp_bias):
        self.temp += new_temp_bias - self._temp_bias
        # update old temp_bias in order that it can be updated again ...
        self._temp_bias = new_temp_bias

    def get_monthly_climate(self, heights, year=None):
        """Monthly climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Returns
        -------
        (temp, tempformelt, prcp, prcpsol)
        """

        y, m = floatyear_to_date(year)
        if self.repeat:
            y = self.ys + (y - self.ys) % (self.ye - self.ys + 1)
        if y < self.ys or y > self.ye:
            raise ValueError('year {} out of the valid time bounds: '
                             '[{}, {}]'.format(y, self.ys, self.ye))
        pok = np.where((self.years == y) & (self.months == m))[0][0]

        # Read already (temperature bias and precipitation factor corrected!)
        itemp = self.temp[pok]
        iprcp = self.prcp[pok]
        igrad = self.grad[pok]

        # For each height pixel:
        # Compute temp and tempformelt (temperature above melting threshold)
        npix = len(heights)
        temp = np.ones(npix) * itemp + igrad * (heights - self.ref_hgt)
        tempformelt = temp - self.t_melt
        clip_min(tempformelt, 0, out=tempformelt)

        # Compute solid precipitation from total precipitation
        prcp = np.ones(npix) * iprcp
        fac = 1 - (temp - self.t_solid) / (self.t_liq - self.t_solid)
        prcpsol = prcp * clip_array(fac, 0, 1)

        return temp, tempformelt, prcp, prcpsol

    def _get_2d_annual_climate(self, heights, year):
        # Avoid code duplication with a getter routine
        year = np.floor(year)
        if self.repeat:
            year = self.ys + (year - self.ys) % (self.ye - self.ys + 1)
        if year < self.ys or year > self.ye:
            raise ValueError('year {} out of the valid time bounds: '
                             '[{}, {}]'.format(year, self.ys, self.ye))
        pok = np.where(self.years == year)[0]
        if len(pok) < 1:
            raise ValueError('Year {} not in record'.format(int(year)))

        # Read already (temperature bias and precipitation factor corrected!)
        itemp = self.temp[pok]
        iprcp = self.prcp[pok]
        igrad = self.grad[pok]

        # For each height pixel:
        # Compute temp and tempformelt (temperature above melting threshold)
        heights = np.asarray(heights)
        npix = len(heights)
        grad_temp = np.atleast_2d(igrad).repeat(npix, 0)
        grad_temp *= (heights.repeat(12).reshape(grad_temp.shape) -
                      self.ref_hgt)
        temp2d = np.atleast_2d(itemp).repeat(npix, 0) + grad_temp
        temp2dformelt = temp2d - self.t_melt
        clip_min(temp2dformelt, 0, out=temp2dformelt)

        # Compute solid precipitation from total precipitation
        prcp = np.atleast_2d(iprcp).repeat(npix, 0)
        fac = 1 - (temp2d - self.t_solid) / (self.t_liq - self.t_solid)
        prcpsol = prcp * clip_array(fac, 0, 1)

        return temp2d, temp2dformelt, prcp, prcpsol

    def get_annual_climate(self, heights, year=None):
        """Annual climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Returns
        -------
        (temp, tempformelt, prcp, prcpsol)
        """
        t, tfmelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)
        return (t.mean(axis=1), tfmelt.sum(axis=1),
                prcp.sum(axis=1), prcpsol.sum(axis=1))

    def get_monthly_mb(self, heights, year=None, **kwargs):

        _, tmelt, _, prcpsol = self.get_monthly_climate(heights, year=year)
        mb_month = prcpsol - self.mu_star * tmelt
        mb_month -= self.bias * SEC_IN_MONTH / SEC_IN_YEAR
        return mb_month / SEC_IN_MONTH / self.rho

    def get_annual_mb(self, heights, year=None, **kwargs):

        _, temp2dformelt, _, prcpsol = self._get_2d_annual_climate(heights,
                                                                   year)
        mb_annual = np.sum(prcpsol - self.mu_star * temp2dformelt, axis=1)
        return (mb_annual - self.bias) / SEC_IN_YEAR / self.rho
    
   
    def get_annual_mb_debris(self, heights, fls=None, year=None, **kwargs):

    
        _, temp2dformelt, _, prcpsol = self._get_2d_annual_climate(heights,
                                                                   year)

        fls_d = self.fls_d 
        d = fls_d.debris_ed # debris enhancement factor
        xx = len(d)
        dd = np.atleast_2d(d).repeat(12,  0)
        c = np.transpose(dd)
        h = np.ones((xx, 12)) 
        v = h * c
        mu_star_debris = v * self.mu_star # elevation-dependent temperature sensitivity parameter with debris enhancement factor
        
                                                                                                                                                                              
        mb_annual = np.sum(prcpsol - mu_star_debris * temp2dformelt, axis=1)                                                                                                               
        return (mb_annual - self.bias) / SEC_IN_YEAR / self.rho

class  MultipleFlowlineMassBalance(MassBalanceModel):
    """Handle mass-balance at the glacier level instead of flowline level.

    Convenience class doing not much more than wrapping a list of mass-balance
    models, one for each flowline.

    This is useful for real-case studies, where each flowline might have a
    different mu*.

    Attributes
    ----------
    fls : list
        list of flowline objects
    mb_models : list
        list of mass-balance objects
    """

    def __init__(self, gdir, fls=None, mu_star=None,
                 mb_model_class=PastMassBalance, use_inversion_flowlines=False,
                 input_filesuffix='', bias=None, **kwargs):
        """Initialize.

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        mu_star : float or list of floats, optional
            set to the alternative value of mu* you want to use
            (the default is to use the calibrated value). Give a list of values
            for flowline-specific mu*
        fls : list, optional
            list of flowline objects to use (defaults to 'model_flowlines',
            and if not available, to 'inversion_flowlines')
        mb_model_class : class, optional
            the mass-balance model to use (e.g. PastMassBalance,
            ConstantMassBalance...)
        use_inversion_flowlines: bool, optional
            if True 'inversion_flowlines' instead of 'model_flowlines' will be
            used.
        input_filesuffix : str
            the file suffix of the input climate file
        bias : float, optional
            set to the alternative value of the calibration bias [mm we yr-1]
            you want to use (the default is to use the calibrated value)
            Note that this bias is *substracted* from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        kwargs : kwargs to pass to mb_model_class
        """

        # Read in the flowlines
        if use_inversion_flowlines:
            fls = gdir.read_pickle('inversion_flowlines')

        if fls is None:
            try:
                fls = gdir.read_pickle('model_flowlines')
            except FileNotFoundError:
                raise InvalidWorkflowError('Need a valid `model_flowlines` '
                                           'file. If you explicitly want to '
                                           'use `inversion_flowlines`, set '
                                           'use_inversion_flowlines=True.')

        self.fls = fls
        _y0 = kwargs.get('y0', None)

        # User mu*?
        if mu_star is not None:
            mu_star = tolist(mu_star, length=len(fls))
            for fl, mu in zip(self.fls, mu_star):
                fl.mu_star = mu

        # Initialise the mb models
        self.flowline_mb_models = []
        for fl in self.fls:
            # Merged glaciers will need different climate files, use filesuffix
            if (fl.rgi_id is not None) and (fl.rgi_id != gdir.rgi_id):
                rgi_filesuffix = '_' + fl.rgi_id + input_filesuffix
            else:
                rgi_filesuffix = input_filesuffix

            # merged glaciers also have a different MB bias from calibration
            if ((bias is None) and cfg.PARAMS['use_bias_for_run'] and
                    (fl.rgi_id != gdir.rgi_id)):
                df = gdir.read_json('local_mustar', filesuffix='_' + fl.rgi_id)
                fl_bias = df['bias']
            else:
                fl_bias = bias

            # Constant and RandomMassBalance need y0 if not provided
            #if (issubclass(mb_model_class, RandomMassBalance) or
               # issubclass(mb_model_class, ConstantMassBalance)) and (
                  #  fl.rgi_id != gdir.rgi_id) and (_y0 is None):

                #df = gdir.read_json('local_mustar', filesuffix='_' + fl.rgi_id)
                #kwargs['y0'] = df['t_star']

            self.flowline_mb_models.append(
                mb_model_class(gdir, mu_star=fl.mu_star, bias=fl_bias,
                               input_filesuffix=rgi_filesuffix, **kwargs))

        self.valid_bounds = self.flowline_mb_models[-1].valid_bounds
        self.hemisphere = gdir.hemisphere

    @property
    def temp_bias(self):
        """Temperature bias to add to the original series."""
        return self.flowline_mb_models[0].temp_bias

    @temp_bias.setter
    def temp_bias(self, value):
        """Temperature bias to add to the original series."""
        for mbmod in self.flowline_mb_models:
            mbmod.temp_bias = value

    @property
    def prcp_fac(self):
        """Precipitation factor to apply to the original series."""
        return self.flowline_mb_models[0].prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, value):
        """Precipitation factor to apply to the original series."""
        for mbmod in self.flowline_mb_models:
            mbmod.prcp_fac = value

    @property
    def bias(self):
        """Residual bias to apply to the original series."""
        return self.flowline_mb_models[0].bias

    @bias.setter
    def bias(self, value):
        """Residual bias to apply to the original series."""
        for mbmod in self.flowline_mb_models:
            mbmod.bias = value

    def get_monthly_mb(self, heights, year=None, fl_id=None, **kwargs):

        if fl_id is None:
            raise ValueError('`fl_id` is required for '
                             'MultipleFlowlineMassBalance!')

        return self.flowline_mb_models[fl_id].get_monthly_mb(heights,
                                                             year=year)

    def get_annual_mb(self, heights, year=None, fl_id=None, **kwargs):

        if fl_id is None:
            raise ValueError('`fl_id` is required for '
                             'MultipleFlowlineMassBalance!')

        return self.flowline_mb_models[fl_id].get_annual_mb(heights,
                                                            year=year)

    def get_annual_mb_debris(self, heights, year=None, fl_id=None, **kwargs):

        if fl_id is None:
            raise ValueError('`fl_id` is required for '
                             'MultipleFlowlineMassBalance!')

        return self.flowline_mb_models[fl_id].get_annual_mb_debris(heights=heights, 
                                                            year=year)

    def get_annual_mb_on_flowlines_debris(self, fls=None, year=None):
        """Get the MB on all points of the glacier at once. (with debris cover)

        Parameters
        ----------
        fls: list, optional
            the list of flowlines to get the mass-balance from. Defaults
            to self.fls
        year: float, optional
            the time (in the "floating year" convention)
        Returns
        -------
        Tuple of (heights, widths, mass_balance) 1D arrays
        """

        if fls is None:
            fls = self.fls

        heights = []
        widths = []
        mbs = []
        for i, fl in enumerate(fls):
            h = fl.surface_h
            heights = np.append(heights, h)
            widths = np.append(widths, fl.widths)
            mbs = np.append(mbs, self.get_annual_mb_debris(h, year=year, fl_id=i))

        return heights, widths, mbs

    def get_annual_mb_on_flowlines(self, fls=None, year=None):
        """Get the MB on all points of the glacier at once.

        Parameters
        ----------
        fls: list, optional
            the list of flowlines to get the mass-balance from. Defaults
            to self.fls
        year: float, optional
            the time (in the "floating year" convention)
        Returns
        -------
        Tuple of (heights, widths, mass_balance) 1D arrays
        """

        if fls is None:
            fls = self.fls

        heights = []
        widths = []
        mbs = []
        for i, fl in enumerate(fls):
            h = fl.surface_h
            heights = np.append(heights, h)
            widths = np.append(widths, fl.widths)
            mbs = np.append(mbs, self.get_annual_mb(h, year=year, fl_id=i))

        return heights, widths, mbs

    def get_specific_mb(self, heights=None, widths=None, fls=None,
                        year=None):

        if heights is not None or widths is not None:
            raise ValueError('`heights` and `widths` kwargs do not work with '
                             'MultipleFlowlineMassBalance!')

        if fls is None:
            fls = self.fls

        if len(np.atleast_1d(year)) > 1:
            out = [self.get_specific_mb(fls=fls, year=yr) for yr in year]
            return np.asarray(out)

        mbs = []
        widths = []
        for i, (fl, mb_mod) in enumerate(zip(self.fls, self.flowline_mb_models)):
            _widths = fl.widths
            try:
                # For rect and parabola don't compute spec mb
                _widths = np.where(fl.thick > 0, _widths, 0)
            except AttributeError:
                pass
            widths = np.append(widths, _widths)
            mb = mb_mod.get_annual_mb(fl.surface_h, year=year, fls=fls, fl_id=i)
            mbs = np.append(mbs, mb * SEC_IN_YEAR * mb_mod.rho)

        return np.average(mbs, weights=widths)

    def get_specific_mb_debris(self, heights=None, widths=None, fls=None,
                        year=None):

        if heights is not None or widths is not None:
            raise ValueError('`heights` and `widths` kwargs do not work with '
                             'MultipleFlowlineMassBalance!')

        if fls is None:
            fls = self.fls

        if len(np.atleast_1d(year)) > 1:
            out = [self.get_specific_mb_debris(fls=fls, year=yr) for yr in year]
            return np.asarray(out)

        mbs = []
        widths = []
        for i, (fl, mb_mod) in enumerate(zip(self.fls, self.flowline_mb_models)):
            _widths = fl.widths
            try:
                # For rect and parabola don't compute spec mb
                _widths = np.where(fl.thick > 0, _widths, 0)
            except AttributeError:
                pass
            widths = np.append(widths, _widths)
            mb = mb_mod.get_annual_mb_debris(fl.surface_h, year=year, fls=fls, fl_id=i)
            mbs = np.append(mbs, mb * SEC_IN_YEAR * mb_mod.rho)

        return np.average(mbs, weights=widths)

    def get_ela(self, year=None, **kwargs):

        # ELA here is not without ambiguity.
        # We compute a mean weighted by area.

        if len(np.atleast_1d(year)) > 1:
            return np.asarray([self.get_ela(year=yr) for yr in year])

        elas = []
        areas = []
        for fl_id, (fl, mb_mod) in enumerate(zip(self.fls,
                                                 self.flowline_mb_models)):
            elas = np.append(elas, mb_mod.get_ela(year=year, fl_id=fl_id,
                                                  fls=self.fls))
            areas = np.append(areas, np.sum(fl.widths))

        return np.average(elas, weights=areas)



@entity_task(log)
def fixed_geometry_mass_balance_debris(gdir, ys=None, ye=None, years=None,
                                monthly_step=False,
                                use_inversion_flowlines=True,
                                climate_filename='climate_historical',
                                climate_input_filesuffix=''):
    """Computes the mass-balance with climate input from e.g. CRU or a GCM.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    ys : int
        start year of the model run (default: from the climate file)
        date)
    ye : int
        end year of the model run (default: from the climate file)
    years : array of ints
        override ys and ye with the years of your choice
    monthly_step : bool
        whether to store the diagnostic data at a monthly time step or not
        (default is yearly)
    use_inversion_flowlines : bool
        whether to use the inversion flowlines or the model flowlines
    climate_filename : str
        name of the climate file, e.g. 'climate_historical' (default) or
        'gcm_data'
    climate_input_filesuffix: str
        filesuffix for the input climate file
    """

    if monthly_step:
        raise NotImplementedError('monthly_step not implemented yet')

    mb = MultipleFlowlineMassBalance(gdir, mb_model_class=PastMassBalance,
                                     filename=climate_filename,
                                     use_inversion_flowlines=use_inversion_flowlines,
                                     input_filesuffix=climate_input_filesuffix)

    if years is None:
        if ys is None:
            ys = mb.flowline_mb_models[0].ys
        if ye is None:
            ye = mb.flowline_mb_models[0].ye
        years = np.arange(ys, ye + 1)

    odf = pd.Series(data=mb.get_specific_mb_debris(year=years),
                    index=years)
    return odf


from oggm.utils._workflow import global_task
@global_task(log)
def compile_fixed_geometry_mass_balance_debris(gdirs, filesuffix='',
                                        path=True, csv=False,
                                        use_inversion_flowlines=True,
                                        ys=None, ye=None, years=None):
    """Compiles a table of specific mass-balance timeseries for all glaciers.

    The file is stored in a hdf file (not csv) per default. Use pd.read_hdf
    to open it.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    filesuffix : str
        add suffix to output file
    path : str, bool
        Set to "True" in order  to store the info in the working directory
        Set to a path to store the file to your chosen location (file
        extension matters)
    csv: bool
        Set to store the data in csv instead of hdf.
    use_inversion_flowlines : bool
        whether to use the inversion flowlines or the model flowlines
    ys : int
        start year of the model run (default: from the climate file)
        date)
    ye : int
        end year of the model run (default: from the climate file)
    years : array of ints
        override ys and ye with the years of your choice
    """
    from oggm.workflow import execute_entity_task
    #from oggm.core.massbalance import fixed_geometry_mass_balance

    out_df = execute_entity_task(fixed_geometry_mass_balance_debris, gdirs,
                                 use_inversion_flowlines=use_inversion_flowlines,
                                 ys=ys, ye=ye, years=years)

    for idx, s in enumerate(out_df):
        if s is None:
            out_df[idx] = pd.Series(np.NaN)

    out = pd.concat(out_df, axis=1, keys=[gd.rgi_id for gd in gdirs])
    out = out.dropna(axis=0, how='all')

    if path:
        if path is True:
            fpath = os.path.join(cfg.PATHS['working_dir'],
                                 'fixed_geometry_mass_balance_debris' + filesuffix)
            if csv:
                out.to_csv(fpath + '.csv')
            else:
                out.to_hdf(fpath + '.hdf', key='df')
        else:
            ext = os.path.splitext(path)[-1]
            if ext.lower() == '.csv':
                out.to_csv(path)
            elif ext.lower() == '.hdf':
                out.to_hdf(path, key='df')
    return out

