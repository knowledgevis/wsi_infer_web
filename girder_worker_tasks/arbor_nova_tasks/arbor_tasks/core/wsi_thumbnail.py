# added for girder interaction as plugin arbor task
from girder_worker.app import app
from girder_worker.utils import girder_job
from tempfile import NamedTemporaryFile

import large_image

#-------------------------------------------

import large_image_source_openslide

@girder_job(title='wsi_thumbnail')
@app.task(bind=True)
def wsi_thumbnail(self,image_file,**kwargs):

    #configure large_image to handle really pig PNGs since sometimes this is used
    large_image.config.setConfig('max_small_image_size',100000)
   
    print('generate a thumbnail for a WSI')
    # open an access handler on the large image.  This will search through all loaded 
    # image sources.  It is a bit more flexible than specifically opening the openslide
    # image source. 
    source = large_image.getTileSource(image_file)

    #print('opening openslilde source by hand')
    #source = large_image_source_openslide.open(image_file)

    # generate unique names for multiple runs.  Add extension so it is easier to use
    outname = NamedTemporaryFile(delete=False).name+'.png'

    thumbnail, mimeType = source.getThumbnail(
        width=800, height=1024, encoding='PNG')
    print('Made a thumbnail of type %s taking %d bytes' % (
        mimeType, len(thumbnail)))
 
    open(outname, 'wb').write(thumbnail)

    print('thumbnail generation complete')
    # return the name of the output file
    return outname

