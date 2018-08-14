"""
DC2 cutout service
"""
import sys
import io
import base64
import argparse
import json

import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import lsst.log as lsstLog
lsstLog.setLevel('CameraMapper', lsstLog.FATAL)
import lsst.daf.persistence as dafPersist
import lsst.afw.geom as afwGeom

from astropy.visualization import ZScaleInterval, make_lupton_rgb
zscale = ZScaleInterval()

_DEFAULT_REPO = '/global/projecta/projectdirs/lsst/global/in2p3/Run1.1/output'
_POSSIBLE_FILTERS = 'ugrizy'


class CutoutMaker():
    """ CutoutMaker class
    """

    def __init__(self, repo=_DEFAULT_REPO, **kwargs):
        self._repo = repo
        self._filter = self._visit_level = None
        self._size = self._image_size = self._pixel_scale = self._side_pixel = None
        self._butler = self._skymap = None
        self._ra = self._dec = self._radec = None
        self._tracts_patches = self._tract = self._patch = self._bbox = None

        if 'ra' in kwargs and 'dec' in kwargs:
            self.coord = kwargs['ra'], kwargs['dec']
        self.visit_level = kwargs.get('visit_level')
        self.filter = kwargs.get('filter', 'gri')
        self.size = kwargs.get('size', 10)  # arcsec
        self.image_size = kwargs.get('image_size', 200)  # pixels
        self.pixel_scale = kwargs.get('pixel_scale', 0.185)  # arcsec per pixel

    @property
    def repo(self):
        """property `repo`: str, stores the repo location"""
        return self._repo

    @property
    def butler(self):
        """property `butler`: dafPersist.Butler instance"""
        if self._butler is None:
            self._butler = dafPersist.Butler(self.repo)
        return self._butler

    @property
    def skymap(self):
        """property `skymap`: sky map"""
        if self._skymap is None:
            self._skymap = self.butler.get('deepCoadd_skyMap')
        return self._skymap

    @property
    def coord(self):
        """property `coord`: tuple, (ra, dec)"""
        return self._ra, self._dec

    @coord.setter
    def coord(self, value):
        ra, dec = value
        ra = float(ra)
        dec = float(dec)
        if self._ra != ra or self._dec != dec:
            self._ra = ra
            self._dec = dec
            self._radec = None

    @property
    def radec(self):
        """property `radec`: afwGeom.SpherePoint instance"""
        if self._radec is None:
            self._radec = afwGeom.SpherePoint(self._ra, self._dec, afwGeom.degrees)
        return self._radec

    @property
    def visit_level(self):
        """property `visit_level`: bool
           when set to True, get cutout from visit level (calexp)
           otherwise, get cutout from coadd (deepCoadd)
        """
        return self._visit_level

    @visit_level.setter
    def visit_level(self, value):
        visit_level = bool(value)
        if self._visit_level != visit_level:
            if visit_level and self.filter and len(self.filter) > 1:
                raise ValueError(
                    '`filter` must be one of "{}*" when `visit_level` set to True'.format(_POSSIBLE_FILTERS))
            self._visit_level = visit_level

    @property
    def size(self):
        """property `size`: float, side length of the cutout in arcsec"""
        return self._size

    @size.setter
    def size(self, value):
        size = float(value)
        if self._size != size:
            self._size = size
            self._side_pixel = None

    @property
    def pixel_scale(self):
        """property `pixel_scale`: float, arcsec per pixel"""
        return self._pixel_scale

    @pixel_scale.setter
    def pixel_scale(self, value):
        pixel_scale = float(value)
        if self._pixel_scale != pixel_scale:
            self._pixel_scale = pixel_scale
            self._side_pixel = None

    @property
    def image_size(self):
        """property `image_size`: int, side length of the cutout in image pixel (output size)"""
        return self._image_size

    @image_size.setter
    def image_size(self, value):
        self._image_size = int(value)

    @property
    def side_pixel(self):
        """property `side_pixel`: int, side length of the cutout in data pixel (array length)"""
        if self._side_pixel is None:
            self._side_pixel = int(round(self._size / self._pixel_scale))
        return self._side_pixel

    @property
    def filter(self):
        """property `filter`: str, name of the filter"""
        return self._filter

    @filter.setter
    def filter(self, value):
        filter = str(value).lower()
        if not ((len(filter) == 1 and filter in _POSSIBLE_FILTERS + '*') or
                (len(filter) == 3 and set(filter).issubset(set(_POSSIBLE_FILTERS)))):
            raise ValueError('"{}" is not a valid filter name!'.format(value))
        if self.visit_level and len(filter) > 1:
            raise ValueError('`filter` must be one of "{}*" when `visit_level` set to True'.format(_POSSIBLE_FILTERS))
        self._filter = filter

    def _iter_tracts_patches(self):
        """returns a list of (tract, patch) for the current options (ra, dec, visit_level)"""
        find_func = getattr(self.skymap, 'findTractPatchList' if self.visit_level else 'findClosestTractPatchList')
        for tract, patches in find_func([self.radec]):
            for patch in patches:
                yield tract, patch
                if not self.visit_level:
                    return

    def _get_bbox(self, tract):
        """returns a afwGeom.BoxI instance given a `tract` and the current options (ra, dec, side_pixel)"""
        xy = afwGeom.PointI(tract.getWcs().skyToPixel(self.radec))
        cutout_size = afwGeom.ExtentI(self.side_pixel, self.side_pixel)
        return afwGeom.BoxI(xy - cutout_size // 2, cutout_size)

    def _fetch_raw_image(self, dataset_type, bbox, data_id):
        """returns the image data array given `dataset_type`, `bbox`, and `data_id`"""
        if not dataset_type.endswith('_sub'):
            dataset_type += '_sub'
        try:
            image = self.butler.get(dataset_type, bbox=bbox, immediate=True, dataId=data_id)
        except dafPersist.NoResults:
            return None
        return image.image.array

    def _iter_dataid_raw_images(self, dataid_only=False, exsiting_only=False, limit=None):
        """iterates over all image data for the current options,
           and yields (dataid, image)
           When a image cannot be retrived, yields (dataid, None)
        """
        count = 0
        for tract, patch in self._iter_tracts_patches():
            bbox = self._get_bbox(tract)
            tract_patch = {'tract': tract.getId(), 'patch': '%d,%d' % patch.getIndex()}
            if self.visit_level:
                partial_dataid = dict(tract_patch)
                if self.filter != '*':
                    partial_dataid['filter'] = self.filter
                for data_ref in self.butler.subset(datasetType='calexp', dataId=partial_dataid):
                    dataid = data_ref.dataId
                    if exsiting_only and not self.butler.datasetExists(datasetType='calexp', dataId=dataid):
                        continue
                    if dataid_only:
                        yield dataid
                    else:
                        yield dataid, self._fetch_raw_image('calexp', bbox, dataid)
                    count += 1
                    if limit and count >= limit:
                        return
            else:
                filters = _POSSIBLE_FILTERS if self.filter == '*' else self.filter
                for filter_this in filters:
                    dataid = dict(tract_patch, filter=filter_this)
                    if exsiting_only and not self.butler.datasetExists(datasetType='deepCoadd', dataId=dataid):
                        continue
                    if dataid_only:
                        yield dataid
                    else:
                        yield dataid, self._fetch_raw_image('deepCoadd', bbox, dataid)
                    count += 1
                    if limit and count >= limit:
                        return

    @property
    def expect_multiple_cutouts(self):
        return self.visit_level or self.filter == '*'

    def format_options(self):
        """Formats current options for printing
        """
        return 'at ({0[0]}, {0[1]}) for filter {1} from {2}'.format(self.coord, self.filter, 'calexp'
                                                                    if self.visit_level else 'deepCoadd')

    def get_images(self, **kwargs):
        """ main method: get cutout images for the given options
            return a list of (dataid, image). If no images found, return an empty list
        """
        if 'ra' in kwargs and 'dec' in kwargs:
            self.coord = kwargs['ra'], kwargs['dec']
        if 'visit_level' in kwargs:
            self.visit_level = kwargs['visit_level']
        if 'filter' in kwargs:
            self.filter = kwargs['filter']
        if 'size' in kwargs:
            self.size = kwargs['size']
        if 'image_size' in kwargs:
            self.image_size = kwargs['image_size']
        if 'pixel_scale' in kwargs:
            self.pixel_scale = kwargs['pixel_scale']

        if len(self.filter) == 3:
            results = list(self._iter_dataid_raw_images(limit=3))
            if len(results) < 3 or all(image is None for _, image in results):
                return []
            zero = np.zeros((self.side_pixel, self.side_pixel))
            images = [zero if image is None else image for _, image in results]
            filters = ''.join(('' if image is None else dataid['filter'] for dataid, image in results))
            dataid = dict(results[0][0], filter=filters)
            return [(dataid, self.image_to_png_data(make_lupton_rgb(*images[::-1], stretch=3, Q=8), composite=True))]

        return [(dataid, self.image_to_png_data(image)) for dataid, image in self._iter_dataid_raw_images(limit=10)
                if image is not None]

    def image_to_png_data(self, image_array, image_size=None, composite=False):
        """Converts image data arry to png data string
        """
        image_size = image_size or self.image_size
        imshow_kwargs = {'origin': 'lower'}
        if not composite:
            imshow_kwargs['cmap'] = 'binary_r'
            imshow_kwargs['vmin'], imshow_kwargs['vmax'] = zscale.get_limits(image_array)
            if not imshow_kwargs['vmin'] <= imshow_kwargs['vmax']:
                del imshow_kwargs['vmin'], imshow_kwargs['vmax']
        fig = plt.figure()
        fig.set_size_inches(1, 1, forward=False)
        ax = plt.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image_array, **imshow_kwargs)
        output = io.BytesIO()
        fig.savefig(output, format='png', dpi=image_size)
        plt.close(fig)
        return 'data:image/png;base64,' + base64.standard_b64encode(output.getvalue()).decode().replace('\n', '')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('coord', nargs='+')
    parser.add_argument('-f', '--filter')
    parser.add_argument('-s', '--size')
    parser.add_argument('-v', '--visit-level', action='store_true')
    parser.add_argument('-r', '--repo')
    parser.add_argument('--pixel-scale')
    parser.add_argument('--image-size')
    parser.add_argument('-k', '--key', help='key for callback service')

    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}

    coord = list(map(float, args['coord']))
    if len(coord) % 2:
        raise ValueError('Must specify both RA and Dec for each object')

    cutout_maker = CutoutMaker(**args)

    output = {
        'key': args.get('key'),
        'one_per_coord': (not cutout_maker.expect_multiple_cutouts),
        'results': list(),
    }

    while coord:
        coord_label = '({}, {})'.format(*coord[:2])
        result = dict(coord=coord_label)
        try:
            data = cutout_maker.get_images(ra=coord[0], dec=coord[1])
        except Exception as e:
            result['error'] = 'Unexpected {}: {}'.format(type(e).__name__, e)
        else:
            if data:
                result['data'] = [{'info': dict(dataid, coord=coord_label), 'image': image} for dataid, image in data]
            else:
                result['error'] = 'Cannot locate image ' + cutout_maker.format_options()
        output['results'].append(result)
        coord = coord[2:]

    json.dump(output, sys.stdout, separators=(',', ':'))
    print()


if __name__ == "__main__":
    main()
