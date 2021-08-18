#!/usr/bin/env python
# -*- coding: utf-8 -*-



from arbor_nova_tasks.arbor_tasks.app_support import infer_wsi 
from arbor_nova_tasks.arbor_tasks.core import wsi_thumbnail

from girder.api import access
from girder.api.describe import Description, autoDescribeRoute
from girder.api.rest import filtermodel, Resource
from girder_worker_utils.transforms.girder_io import GirderFileId, GirderUploadToItem


class ArborNova(Resource):
    def __init__(self):
        super(ArborNova, self).__init__()
        self.resourceName = 'arbor_nova'
        self.route('POST', ('infer_wsi', ), self.infer_wsi)
        self.route('POST', ('wsi_thumbnail', ), self.wsi_thumbnail)
    @access.token
    @filtermodel(model='job', plugin='jobs')



    # ---DNN infer command line for NIH AI resource group algorithm
 
    @access.token
    @filtermodel(model='job', plugin='jobs')
    @autoDescribeRoute(
        Description('perform forward inferencing using a pretrained network')
        .param('imageId', 'The ID of the source, an Aperio .SVS image file.')
        .param('outputId', 'The ID of the output item where the output file will be uploaded.')
        .param('statsId', 'The ID of the output item where the output file will be uploaded.')
        .errorResponse()
        .errorResponse('Write access was denied on the parent item.', 403)
        .errorResponse('Failed to upload output file.', 500)
    )
    def infer_wsi(
            self, 
            imageId, 
            outputId,
            statsId
    ):  
        result = infer_wsi.delay(
                GirderFileId(imageId), 
                girder_result_hooks=[
                    GirderUploadToItem(outputId),
                    GirderUploadToItem(statsId),
                ])
        return result.job

    # --- generate a thumbnail from a pyramidal image
    @access.token
    @filtermodel(model='job', plugin='jobs')
    @autoDescribeRoute(
        Description('generate a wsi_thumbnail')
        .param('imageId', 'The ID of the source, an Aperio .SVS image file.')
        .param('outputId', 'The ID of the output item where the output file will be uploaded.')
        .errorResponse()
        .errorResponse('Write access was denied on the parent item.', 403)
        .errorResponse('Failed to upload output file.', 500)
    )
    def wsi_thumbnail(
            self, 
            imageId, 
            outputId
    ):
        result = wsi_thumbnail.delay(
                GirderFileId(imageId), 
                girder_result_hooks=[
                    GirderUploadToItem(outputId)
                ])
        return result.job
