"""
DC2 cutout service
"""
import sys
import io
import base64

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import lsst.log as lsstLog
lsstLog.setLevel('CameraMapper', lsstLog.FATAL)
import lsst.daf.persistence as dafPersist
import lsst.afw.geom as afwGeom

from astropy.visualization import ZScaleInterval, make_lupton_rgb
zscale = ZScaleInterval()

_pixel_scale = 0.185
_default_repo = '/global/projecta/projectdirs/lsst/global/in2p3/Run1.1/output'
_image_size = 250

def make_cutout(ra, dec, filters='r', side_arcsec=10, datasetType='deepCoadd',
                image_size=_image_size, repo=_default_repo, 
                pixel_scale=_pixel_scale):
    
    if len(filters) not in (1, 3):
        raise ValueError('`filters` must have length 1 or 3')

    butler = dafPersist.Butler(repo)
    skymap = butler.get("%s_skyMap" % datasetType)

    radec = afwGeom.SpherePoint(float(ra), float(dec), afwGeom.degrees)
    tractInfo = skymap.findTract(radec)
    patchInfo = tractInfo.findPatch(radec)

    xy = afwGeom.PointI(tractInfo.getWcs().skyToPixel(radec))
    cutoutSideLength = int(round(float(side_arcsec) / pixel_scale))
    cutoutSize = afwGeom.ExtentI(cutoutSideLength, cutoutSideLength)
    bbox = afwGeom.BoxI(xy - cutoutSize//2, cutoutSize)

    images = []
    for filter_this in filters:
        coaddId = {'tract': tractInfo.getId(), 'patch': '%d,%d'%patchInfo.getIndex(), 'filter': filter_this}
        cutout_image = butler.get(datasetType+'_sub', bbox=bbox, immediate=True, dataId=coaddId)
        images.append(cutout_image.image.array)

    if len(images) == 3:
        images = make_lupton_rgb(*images[::-1], stretch=3, Q=8)
        is_rgb = True
    else:
        images = images.pop()
        is_rgb = False

    return generate_png_data(images, int(image_size), is_rgb)

def generate_png_data(cutout_image_array, size=_image_size, is_rgb=False):
    imshow_kwargs = {'origin': 'lower'}
    if not is_rgb:
        imshow_kwargs['cmap'] = 'binary_r'
        imshow_kwargs['vmin'], imshow_kwargs['vmax'] = zscale.get_limits(cutout_image_array)
    fig = plt.figure()
    fig.set_size_inches(1, 1, forward=False)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(cutout_image_array, **imshow_kwargs)
    output = io.BytesIO()
    fig.savefig(output, format='png', dpi=size)
    plt.close(fig)
    return base64.standard_b64encode(output.getvalue()).decode().replace('\n', '')


def main():
    print('data:image/png;base64,' + make_cutout(*sys.argv[1:]))
