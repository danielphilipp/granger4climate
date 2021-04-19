"""
Main script containing code to apply Granger causality tests to

(1) 1D timeseries of two variables
(2) 3D gridded data with shape (time, lat, lon) of two variables

"""

import xarray as xr
import numpy as np
import numpy.ma as ma
import os
from statsmodels.tsa.stattools import adfuller,grangercausalitytests
from statsmodels.tsa.vector_ar import var_model as vm
from scipy.stats import pearsonr
import matplotlib.patches as mpatch
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf



class G4C:
    def __init__(self, x, y, xvar, yvar):
        """  """
        self.x = x
        self.y = y
        self.xvar = xvar
        self.yvar = yvar

        self._get_is_2d()

        self.spl = None
        self.maxlag = None
        self.ic_method=None
        self.k_ar = None
        self.x_causing_y = None
        self.y_causing_x = None
        self.latdim = None
        self.londim = None
        self.x_prepared = None
        self.y_prepared = None
        self.ds = None
        self.do_correlation = None
        self.cor_p = None
        self.cor_r = None

    def _get_is_2d(self):
        """ Check if data is spatially averaged. """
        ndimsx = len(self.x.shape)
        ndimsy = len(self.y.shape)

        assert ndimsx == ndimsy

        if ndimsx == 3:
            self.is_2d = True
        elif ndimsx == 1:
            self.is_2d = False
        else:
            raise Exception('Input data must have 1 or 3 dimensions!')

    def _deseasonalize(self, x, y):
        """ Deseasonalize data. """
        xvals = ma.empty(x.shape)
        yvals = ma.empty(y.shape)

        for m in range(0, 12):
            if self.is_2d:
                xmean = ma.mean(x[m::12, :, :], axis=0)
                ymean = ma.mean(y[m::12, :, :], axis=0)
                xvals[m::12, :, :] = x[m::12, :, :] - xmean
                yvals[m::12, :, :] = y[m::12, :, :] - ymean
            else:
                xmean = ma.mean(x[m::12], axis=0)
                ymean = ma.mean(y[m::12], axis=0)
                xvals[m::12] = x[m::12] - xmean
                yvals[m::12] = y[m::12] - ymean
        return xvals, yvals

    def _standardize(self, data):
        """ Standardize timeseries for zero mean and unit variance """
        return (data - ma.mean(data)) / ma.std(data)

    def _stationarity(self, x, y):
        """
        """

        order = 1
        error = False
        try:
            p_value_x = adfuller(x,
                                 maxlag=None,
                                 regression='c',
                                 autolag='BIC')[1]
        except:
            error = True
        try:
            p_value_y = adfuller(y,
                                 maxlag=None,
                                 regression='c',
                                 autolag='BIC')[1]
        except:
            error = True

        if not error:
            if (p_value_x > self.spl or p_value_y > self.spl):
                if not self.is_2d:
                    print('Timeseries not stationary. Applying 1st '
                          'order differencing\n')
                xdiff = list()
                ydiff = list()
                for i in range(order, len(x)):
                    xdiff.append(x[i] - x[i - order])
                    ydiff.append(y[i] - y[i - order])

                order += 1
                x = np.asarray(xdiff)
                y = np.asarray(ydiff)

                p_value_x = adfuller(x,
                                     maxlag=None,
                                     regression='c',
                                     autolag='BIC')[1]
                p_value_y = adfuller(y,
                                     maxlag=None,
                                     regression='c',
                                     autolag='BIC')[1]

                if (p_value_x <= self.spl and p_value_y <= self.spl):
                    if not self.is_2d:
                        print('Both timeseries stationary after 1st order '
                              'differencing with pval_lim={}'.format(self.spl))
                else:
                    if not self.is_2d:
                        print('One or both timeseries still not stationary after '
                              '1st order differencing '
                              'with pval_lim={}'.format(self.spl))
                if not self.is_2d:
                    print('Adfuller x p-value after 1st order '
                          'differencing: {:.3f}'.format(p_value_x))
                    print('Adfuller y p-value after 1st order '
                          'differencing: {:.3f}\n'.format(p_value_y))

                return np.asarray(x), np.asarray(y)
            else:
                return x, y
        else:
            return ma.masked_all(x.shape), ma.masked_all(y.shape)

    def _correlate(self, x, y):
        return pearsonr(x, y)

    def run_causalitytest(self, maxlag=15, stationarity_pval_lim=0.05,
                 ic_method='bic', do_correlation=True):

        self.maxlag = maxlag
        self.spl = stationarity_pval_lim
        self.ic_method = ic_method
        self.do_correlation = do_correlation

        x = self.x
        y = self.y

        x, y = self._deseasonalize(x, y)
        x = ma.masked_invalid(x)
        y = ma.masked_invalid(y)

        if self.is_2d:
            latdim = x.shape[1]
            londim = y.shape[2]

            self.latdim = latdim
            self.londim = londim

            y_causing_x = ma.empty((self.maxlag, 2, latdim, londim))
            x_causing_y = ma.empty((self.maxlag, 2, latdim, londim))

            cor_r = ma.empty((latdim, londim))
            cor_p = ma.empty((latdim, londim))

            for i in range(latdim):
                for j in range(londim):
                    xtmp = x[:, i, j]
                    ytmp = y[:, i, j]

                    has_invalid_x = ma.count_masked(xtmp) > 0
                    has_invalid_y = ma.count_masked(ytmp) > 0

                    if not has_invalid_x and not has_invalid_y:

                        xtmp, ytmp = self._stationarity(xtmp, ytmp)
                        xtmp = ma.masked_invalid(xtmp)
                        ytmp = ma.masked_invalid(ytmp)
                        xtmp = self._standardize(xtmp)
                        ytmp = self._standardize(ytmp)
                        xtmp = ma.masked_invalid(xtmp)
                        ytmp = ma.masked_invalid(ytmp)

                        # f_value_01, p_value_01, f_value_10, p_value_10
                        res = self._gc_test(xtmp, ytmp)

                        if self.do_correlation:
                            cor_r[i, j], cor_p[i, j] = self._correlate(xtmp, ytmp)

                        for lag in range(maxlag-1):
                            x_causing_y[lag, 0, i, j] = res[2][lag]
                            x_causing_y[lag, 1, i, j] = res[3][lag]
                            y_causing_x[lag, 0, i, j] = res[0][lag]
                            y_causing_x[lag, 1, i, j] = res[1][lag]
                    else:
                        for lag in range(maxlag - 1):
                            x_causing_y[lag, 0, i, j] = ma.masked
                            x_causing_y[lag, 1, i, j] = ma.masked
                            y_causing_x[lag, 0, i, j] = ma.masked
                            y_causing_x[lag, 1, i, j] = ma.masked
                        cor_r[i,j] = ma.masked
                        cor_p[i, j] = ma.masked

            x_causing_y = np.squeeze(ma.masked_invalid(x_causing_y))
            y_causing_x = np.squeeze(ma.masked_invalid(y_causing_x))
            x_causing_y = np.squeeze(ma.masked_outside(x_causing_y, -500, 500))
            y_causing_x = np.squeeze(ma.masked_outside(y_causing_x, -500, 500))

        else:
            y_causing_x = ma.empty((self.maxlag, 2))
            x_causing_y = ma.empty((self.maxlag, 2))

            x, y = self._stationarity(x, y)
            x = ma.masked_invalid(x)
            y = ma.masked_invalid(y)
            x = self._standardize(x)
            y = self._standardize(y)
            x = ma.masked_invalid(x)
            y = ma.masked_invalid(y)
            # f_value_01, p_value_01, f_value_10, p_value_10
            res = self._gc_test(x, y)
            self.k_ar = self._get_opt_model_order(x, y)

            if self.do_correlation:
                cor_r, cor_p = self._correlate(x, y)

            for lag in range(maxlag-1):
                x_causing_y[lag, 0] = res[2][lag]
                x_causing_y[lag, 1] = res[3][lag]
                y_causing_x[lag, 0] = res[0][lag]
                y_causing_x[lag, 1] = res[1][lag]

        self.x_causing_y = x_causing_y
        self.y_causing_x = y_causing_x

        self.x_prepared = x
        self.y_prepared = y

        if self.do_correlation:
            self.cor_r = cor_r
            self.cor_p = cor_p

    def save_results(self, opath):
        if self.is_2d:
            ndim = '3D'

            xy_F = xr.DataArray(data=self.x_causing_y[:, 0, :, :],
                                dims=['lag', 'lat', 'lon'])
            xy_p = xr.DataArray(data=self.x_causing_y[:, 1, :, :],
                                dims=['lag', 'lat', 'lon'])
            yx_F = xr.DataArray(data=self.y_causing_x[:, 0, :, :],
                                dims=['lag', 'lat', 'lon'])
            yx_p = xr.DataArray(data=self.y_causing_x[:, 1, :, :],
                                dims=['lag', 'lat', 'lon'])
            xarr = xr.DataArray(data=self.x,
                                dims=['time', 'lat', 'lon'])
            yarr = xr.DataArray(data=self.y,
                                dims=['time', 'lat', 'lon'])
            xarr_prep = xr.DataArray(data=self.x_prepared,
                                     dims=['time_prep', 'lat', 'lon'])
            yarr_prep = xr.DataArray(data=self.y_prepared,
                                dims=['time_prep', 'lat', 'lon'])
            if self.do_correlation:
                cor_r = xr.DataArray(data=self.cor_r,
                                     dims=['lat', 'lon'])
                cor_p = xr.DataArray(data=self.cor_p,
                                     dims=['lat', 'lon'])
        else:
            ndim = '1D'

            xy_F = xr.DataArray(data=self.x_causing_y[:, 0],
                                dims=['lag'])
            xy_p = xr.DataArray(data=self.x_causing_y[:, 1],
                                dims=['lag'])
            yx_F = xr.DataArray(data=self.y_causing_x[:, 0],
                                dims=['lag'])
            yx_p = xr.DataArray(data=self.y_causing_x[:, 1],
                                dims=['lag'])
            xarr = xr.DataArray(data=self.x,
                                dims=['time'])
            yarr = xr.DataArray(data=self.y,
                                dims=['time'])
            xarr_prep = xr.DataArray(data=self.x_prepared,
                                     dims=['time_prep'])
            yarr_prep = xr.DataArray(data=self.y_prepared,
                                dims=['time_prep'])
            k_ar = xr.DataArray(data=np.array([self.k_ar]), dims=['idx'])

            if self.do_correlation:

                cor_r = xr.DataArray(data=np.array([self.cor_r]),
                                     dims=['idx'])
                cor_p = xr.DataArray(data=np.array([self.cor_p]),
                                     dims=['idx'])

        filename = 'GC_{}_{}_maxlag-{}_{}.nc'
        filename = filename.format(self.xvar, self.yvar, self.maxlag, ndim)

        ptf = os.path.join(opath, filename)

        ds = xr.Dataset()
        ds['{}_{}_Fstat'.format(self.xvar, self.yvar)] = xy_F
        ds['{}_{}_pval'.format(self.xvar, self.yvar)] = xy_p
        ds['{}_{}_Fstat'.format(self.yvar, self.xvar)] = yx_F
        ds['{}_{}_pval'.format(self.yvar, self.xvar)] = yx_p
        ds['x_{}'.format(self.xvar)] = xarr
        ds['y_{}'.format(self.yvar)] = yarr
        ds['x_{}_prep'.format(self.xvar)] = xarr_prep
        ds['y_{}_prep'.format(self.yvar)] = yarr_prep
        ds['cor_r'] = cor_r
        ds['cor_p'] = cor_p
        if not self.is_2d:
            ds['opt_model_order'] = k_ar

        enc = {}
        for v in ds.variables:
            if v == 'k_ar':
                enc[v] = {'dtype': np.int16}
            else:
                enc[v] = {'dtype': np.float32, '_FillValue': -999.}

        ds.attrs['xvar'] = self.xvar
        ds.attrs['yvar'] = self.yvar

        ds.to_netcdf(ptf, encoding=enc)

        print('Results saved: \n')
        print(ds)

        self.ds = ds

    def _gc_test(self, x, y):
        """
        """
        data1 = np.column_stack((np.asarray(x), np.asarray(y)))
        data2 = np.column_stack((np.asarray(y), np.asarray(x)))

        err1 = False
        err2 = False

        try:
            gct1 = grangercausalitytests(data1,
                                         maxlag=self.maxlag, verbose=False)
        except:
            err1 = True

        try:
            gct2 = grangercausalitytests(data2,
                                         maxlag=self.maxlag, verbose=False)
        except:
            err2 = True

        if not err1:
            F_stat1_ret = []
            p_val1_ret = []
            for i in range(1, self.maxlag):
                F_stat1_ret.append(gct1[i][0]["ssr_ftest"][0])
                p_val1_ret.append(gct1[i][0]["ssr_ftest"][1])
        else:
            F_stat1_ret = [ma.masked] * self.maxlag
            p_val1_ret = [ma.masked] * self.maxlag

        if not err2:
            F_stat2_ret = []
            p_val2_ret = []
            for i in range(1, self.maxlag):
                F_stat2_ret.append(gct2[i][0]["ssr_ftest"][0])
                p_val2_ret.append(gct2[i][0]["ssr_ftest"][1])
        else:
            F_stat2_ret = [ma.masked] * self.maxlag
            p_val2_ret = [ma.masked] * self.maxlag

        return F_stat1_ret, p_val1_ret, F_stat2_ret, p_val2_ret

    def _get_opt_model_order(self, x, y):
        ts = np.column_stack((x, y))
        VAR_model = vm.VAR(ts)
        results = VAR_model.fit(ic=self.ic_method, maxlags=30, verbose=False)
        return results.k_ar


class G4CPlotting:
    def __init__(self, idata):
        self.idata = idata
        if isinstance(idata, str):
            if not os.path.isfile(idata):
                raise Exception('File {} does not exist!'.format(idata))
            self.ds = xr.open_dataset(idata)
        elif isinstance(idata, G4C):
            self.ds = idata.ds
        else:
            raise Exception('idata has to be instance of G4C or netcdf4 file'
                            'written by G4C!')

        self.nlags = self.ds.dims['lag']
        self.xvar = self.ds.xvar
        self.yvar = self.ds.yvar

    def plot_1d_lags(self, figname=None, Fyaxis_max=14):

        avail_Fstats = ['{}_{}_Fstat'.format(self.xvar, self.yvar),
                        '{}_{}_Fstat'.format(self.yvar, self.xvar)]

        avail_pvals = ['{}_{}_pval'.format(self.xvar, self.yvar),
                       '{}_{}_pval'.format(self.yvar, self.xvar)]

        opt_model_order = self.ds['opt_model_order'][0]
        x = np.arange(1, self.nlags+1)

        assert len(avail_Fstats) == 2
        assert len(avail_pvals) == 2

        fig = plt.figure(figsize=(14, 5))

        ax = fig.add_subplot(111)
        ax.grid(linestyle="--", color="grey")

        plt1 = ax.plot(x, self.ds[avail_Fstats[0]],
                       color="blue", label=avail_Fstats[0], marker="o", ls="-",
                       markersize=6, linewidth=3)
        plt2 = ax.plot(x, self.ds[avail_Fstats[1]],
                       color="blue", label=avail_Fstats[1], marker="o", ls=":",
                       markersize=6, linewidth=3)

        ax.set_ylim(0, Fyaxis_max)
        ax.set_xticks(np.arange(1, self.nlags+1)[::1])
        ax.set_xlabel("Model order", fontsize=14)
        ax.set_ylabel("F-statistic", color="blue", fontsize=14)
        ax.tick_params('y', colors='blue')
        ax.tick_params(labelsize=14)
        ax.set_xlim(0, self.nlags)

        ax1 = ax.twinx()
        ax1.add_patch(
            mpatch.Rectangle((0, 0), self.nlags, 0.05,
                             color="red", alpha=0.15))
        plt3 = ax1.plot(x, self.ds[avail_pvals[0]],
                        color="r", label=avail_pvals[0], marker="^", ls="-",
                        markersize=6, linewidth=3)
        plt4 = ax1.plot(x, self.ds[avail_pvals[1]],
                        color="r", label=avail_pvals[1], marker="^", ls=":",
                        markersize=6, linewidth=3)
        ax1.set_ylabel('p-value', color='r', fontsize=14)
        ax1.tick_params('y', colors='r')
        ax1.set_ylim(0, .1)
        # ax1.axhline(y=0.05,color="r",linestyle=":",linewidth=3)
        ax1.tick_params(labelsize=14)

        ax1.annotate(" ", xy=(opt_model_order, 0),
                     xytext=(opt_model_order, -0.01),
                     arrowprops=dict(facecolor="black"))

        plts = plt1 + plt2 + plt3 + plt4
        labs = [l.get_label() for l in plts]

        plt.legend(plts, labs, fontsize=10, loc=9)
        plt.tight_layout()
        if figname is None:
            plt.show()
        else:
            plt.savefig(figname)

    def plot_1d_auxdata(self, figname=None):

        fig = plt.figure(figsize=(14,6))

        ax = fig.add_subplot(2,2,1)
        v = 'x_' + self.xvar
        ax.plot(self.ds[v])
        ax.grid(color='grey', linestyle='--')
        ax.set_title(v, fontweight='bold')

        ax = fig.add_subplot(2,2,2)
        v = 'y_' + self.yvar
        ax.plot(self.ds[v])
        ax.grid(color='grey', linestyle='--')
        ax.set_title(v, fontweight='bold')

        ax = fig.add_subplot(2,2,3)
        v = 'x_' + self.xvar + '_prep'
        ax.plot(self.ds[v], color='red')
        ax.set_title(v)
        ax.axhline(y=0, color='black')
        ax.grid(color='grey', linestyle='--')
        ax.set_title(v, fontweight='bold')

        ax = fig.add_subplot(2,2,4)
        v = 'y_' + self.yvar + '_prep'
        ax.plot(self.ds[v], color='red')
        ax.set_title(v)
        ax.axhline(y=0, color='black')
        ax.grid(color='grey', linestyle='--')
        ax.set_title(v, fontweight='bold')

        plt.tight_layout()
        if figname is None:
            plt.show()
        else:
            plt.savefig(figname)

    def _circle_bounds(self):
            """
            """
            theta = np.linspace(0, 2*np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)
            return circle

    def plot_2d_maps(self, orders, proj_params, figname=None, masking=False,
                     pval_lim=0.05, lims=None):

        if lims is None:
            lims = [(None, None)] * len(orders)

        ds = self.ds

        Fxy = '{}_{}_Fstat'.format(self.xvar, self.yvar)
        Fyx = '{}_{}_Fstat'.format(self.yvar, self.xvar)

        pxy = '{}_{}_pval'.format(self.xvar, self.yvar)
        pyx = '{}_{}_pval'.format(self.yvar, self.xvar)

        if masking:
            ds[Fxy] = xr.where(ds[pxy] > pval_lim, np.nan, ds[Fxy])
            ds[Fyx] = xr.where(ds[pyx] > pval_lim, np.nan, ds[Fyx])

        if isinstance(proj_params['oproj'], ccrs.NorthPolarStereo):
            circle = self._circle_bounds()

        cmap = plt.get_cmap("YlOrRd")

        fig = plt.figure()
        for cnt, order in enumerate(orders):

            ax = fig.add_subplot(2, len(orders), cnt+1,
                                 projection=proj_params['oproj'])

            ax.set_extent(proj_params['extent'], crs=proj_params['iproj'])

            if isinstance(proj_params['oproj'], ccrs.NorthPolarStereo):
                ax.set_boundary(circle, transform=ax.transAxes)

            ax.set_title(Fyx + ' | Order: {}'.format(order), fontsize=16,
                         fontweight="bold")
            ax.add_feature(cf.LAND, color="darkgray")
            ax.coastlines(resolution="50m")
            ax.gridlines(linestyle=":", alpha=0.5, color="black")


            ims = ax.imshow(ds[Fxy][order-1, :, :], origin='lower',
                            extent=proj_params['extent'],
                            cmap=cmap,
                            transform=proj_params['iproj'],
                            vmin=lims[cnt][0],
                            vmax=lims[cnt][1])

            cb = plt.colorbar(ims)
            cb.set_label("F-Statistic", size=14)

        for cnt, order in enumerate(orders):

            ax = fig.add_subplot(2, len(orders), len(orders) + cnt+1,
                                 projection=proj_params['oproj'])

            ax.set_extent(proj_params['extent'], crs=proj_params['iproj'])

            if isinstance(proj_params['oproj'], ccrs.NorthPolarStereo):
                ax.set_boundary(circle, transform=ax.transAxes)

            ax.set_title(Fyx + ' | Order: {}'.format(order), fontsize=16,
                         fontweight="bold")
            ax.add_feature(cf.LAND, color="darkgray")
            ax.coastlines(resolution="50m")
            ax.gridlines(linestyle=":", alpha=0.5, color="black")

            ims = ax.imshow(ds[Fyx][order-1, :, :],
                            extent=proj_params['extent'],
                            cmap=cmap,
                            transform=proj_params['iproj'],
                            vmin=lims[cnt][0],
                            vmax=lims[cnt][1])
            cb = plt.colorbar(ims)
            cb.set_label("F-Statistic", size=14)

        plt.show()
