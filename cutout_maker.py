"""
DC2 cutout service
"""
import sys
import io
import base64
import argparse

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import lsst.log as lsstLog
lsstLog.setLevel('CameraMapper', lsstLog.FATAL)
import lsst.daf.persistence as dafPersist
import lsst.afw.geom as afwGeom

from astropy.visualization import ZScaleInterval, make_lupton_rgb
zscale = ZScaleInterval()

class CutoutMaker():
    def __init__(self, **kwargs):
        self._repo = kwargs.get('repo', '/global/projecta/projectdirs/lsst/global/in2p3/Run1.1/output')
        self._filter = kwargs.get('filter', 'gri')
        self._dataset_type = kwargs.get('dataset_type', 'deepCoadd')
        self._size = float(kwargs.get('size', 10.0)) # arcsec
        self._image_size = int(kwargs.get('image_size', 200)) # pixels
        self._pixel_scale = float(kwargs.get('pixel_scale', 0.185)) # arcsec per pixcel

        self._ra = self._dec = self._radec = None
        self._tract = self._patch = self._bbox = self._side_pixel = None
        self._butler = self._skymap = None

    @property
    def coord(self):
        return self._ra, self._dec

    @coord.setter
    def coord(self, value):
        ra, dec = value
        ra = float(ra)
        dec = float(dec)
        if self._ra != ra or self._dec != dec:
            self._ra = ra
            self._dec = dec
            self.radec = None

    @property
    def filter(self):
        return self._filter

    @filter.setter
    def filter(self, value):
        if len(value) not in (1, 3):
            raise ValueError('`filters` must have length 1 or 3')
        self._filter = value

    @property
    def repo(self):
        return self._repo

    @repo.setter
    def repo(self, value):
        if self._repo != value:
            self._repo = value
            self.butler = None

    @property
    def dataset_type(self):
        return self._dataset_type

    @dataset_type.setter
    def dataset_type(self, value):
        if self._dataset_type != value:
            self._dataset_type = value
            self.skymap = None

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        size = float(value)
        if self._size != size:
            self._size = size
            self.side_pixel = None

    @property
    def pixel_scale(self):
        return self._pixel_scale

    @pixel_scale.setter
    def pixel_scale(self, value):
        pixel_scale = float(value)
        if self._pixel_scale != pixel_scale:
            self._pixel_scale = pixel_scale
            self.side_pixel = None

    @property
    def image_size(self):
        return self._image_size

    @image_size.setter
    def image_size(self, value):
        self._image_size = int(value)

    @property
    def butler(self):
        if self._butler is None:
            self._butler = dafPersist.Butler(self.repo)
        return self._butler

    @butler.setter
    def butler(self, value):
        self._butler = value
        self.skymap = None

    @property
    def skymap(self):
        if self._skymap is None:
            self._skymap = self.butler.get("%s_skyMap" % self.dataset_type)
        return self._skymap

    @skymap.setter
    def skymap(self, value):
        self._skymap = value
        self._tract = None

    @property
    def radec(self):
        if self._radec is None:
            self._radec = afwGeom.SpherePoint(self._ra, self._dec, afwGeom.degrees)
        return self._radec

    @radec.setter
    def radec(self, value):
        self._radec = value
        self.tract = None
        self.patch = None
        self.bbox = None

    @property
    def tract(self):
        if self._tract is None:
            self._tract = self.skymap.findTract(self.radec)
        return self._tract

    @tract.setter
    def tract(self, value):
        self._tract = value
        self.patch = None
        self.bbox = None

    @property
    def patch(self):
        if self._patch is None:
            self._patch = self._tract.findPatch(self.radec)
        return self._patch

    @patch.setter
    def patch(self, value):
        self._patch = value
        self.bbox = None

    @property
    def bbox(self):
        if self._bbox is None:
            xy = afwGeom.PointI(self.tract.getWcs().skyToPixel(self.radec))
            cutout_size = afwGeom.ExtentI(self.side_pixel, self.side_pixel)
            self._bbox = afwGeom.BoxI(xy - cutout_size//2, cutout_size)
        return self._bbox

    @bbox.setter
    def bbox(self, value):
        self._bbox = value

    @property
    def side_pixel(self):
        if self._side_pixel is None:
            self._side_pixel = int(round(self._size / self._pixel_scale))
        return self._side_pixel

    @side_pixel.setter
    def side_pixel(self, value):
        self._side_pixel = value
        self.bbox = None

    def get_image(self):
        if len(self.filter) == 3:
            filters = self.filter
            images = []
            for filter_this in filters:
                self.filter = filter_this
                images.append(self.get_image())
            self.filter = filters
            return make_lupton_rgb(*images[::-1], stretch=3, Q=8)
        coaddId = {'tract': self.tract.getId(), 'patch': '%d,%d'%self.patch.getIndex(), 'filter': self.filter}
        return self.butler.get(self.dataset_type+'_sub', bbox=self.bbox, immediate=True, dataId=coaddId).image.array

    def get_png_data(self):
        cutout_image_array = self.get_image()
        imshow_kwargs = {'origin': 'lower'}
        if len(self.filter) == 1:
            imshow_kwargs['cmap'] = 'binary_r'
            imshow_kwargs['vmin'], imshow_kwargs['vmax'] = zscale.get_limits(cutout_image_array)
        fig = plt.figure()
        fig.set_size_inches(1, 1, forward=False)
        ax = plt.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(cutout_image_array, **imshow_kwargs)
        output = io.BytesIO()
        fig.savefig(output, format='png', dpi=self.image_size)
        plt.close(fig)
        return 'data:image/png;base64,' + base64.standard_b64encode(output.getvalue()).decode().replace('\n', '')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('coord', nargs='+')
    parser.add_argument('-f', '--filter')
    parser.add_argument('-s', '--size')
    parser.add_argument('--repo')
    parser.add_argument('--dataset-type')
    parser.add_argument('--pixel-scale')
    parser.add_argument('--image-size')

    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}

    coord = args['coord']
    if len(coord) % 2:
        raise ValueError('Must specify both RA and Dec for each object')
    del args['coord']

    cutout_maker = CutoutMaker(**args)
    while coord:
        cutout_maker.coord = coord[:2]
        print(cutout_maker.get_png_data())
        coord = coord[2:]

if __name__ == "__main__":
    main()
