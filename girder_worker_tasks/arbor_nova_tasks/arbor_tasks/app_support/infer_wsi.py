# added for girder interaction as plugin arbor task
from girder_worker.app import app
from girder_worker.utils import girder_job
from tempfile import NamedTemporaryFile

# try to keep fastai from spawning multiple processes
import os
os.sched_setaffinity(0, (0,))

# declared for subprocess to do GPU stuff.  Package 'billiard' comes with celery
# and is a workaround for subprocess limitations on 'daemonic' processes.

#subprocessMethod = 'billiard'
#subprocessMethod = 'torch'
subprocessMethod = 'none'

if (subprocessMethod == "torch"):
    print('setup torch multiprocessing')
    import torch.multiprocessing as multiprocessing
    #import billiard as multiprocessing
    from torch.multiprocessing import Queue, Process
    from torch.multiprocessing import set_start_method
elif (subprocessMethod == "billiard"):
    print('setup billiard multiprocessing')
    import billiard
    import billiard as multiprocessing
    from billiard import Queue, Process
else:
    print('no subprocess method set')

import json



# path to model file may vary depending on whether this is in a docker container.  We could make
# this more elegant using environment variables, but simple is also good
#modelPath = '/models/deeplabv3_resnet50_10ep_lr1e4_nonorm.pkl' 
modelPath = './models/deeplabv3_resnet50_10ep_lr1e4_nonorm.pkl'

#-------------------------------------------
import sys
import os
import numpy as np
import cv2
import openslide
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import xml.etree.ElementTree as ET
from xml.dom import minidom
import geojson
import argparse
from fastai.vision.all import *
import matplotlib.pyplot as plt
import fastai
import PIL
matplotlib.use('Agg')
import pandas as pd
import datetime
from skimage import draw, measure, morphology
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon, shape
from shapely.ops import cascaded_union, unary_union
import json
import shapely
import warnings
warnings.filterwarnings("ignore")
#-------------------------------------------

@girder_job(title='inferWSI')
@app.task(bind=True)
def infer_wsi(self,image_file,**kwargs):

    print(" input image filename = {}".format(image_file))

    # setup the GPU environment for pytorch
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    DEVICE = torch.device(f"cuda:0")

    print('perform forward inferencing')

    # generate a spot in Girder for the output data file (a geoJSON file)
    # We generate unique names for multiple runs.  
    outname = NamedTemporaryFile(delete=False).name+'_preds.json'

    if (subprocessMethod == 'torch'):
        # declare a subprocess that does the GPU allocation to keep the GPU memory from leaking
        print('starting a torch subprocess')
        set_start_method("spawn")
        msg_queue = Queue()
        #gpu_process = torch.multiprocessing.spawn(fn=runInference_Subprocess, 
        #               args=(msg_queue,image_file,outname,modelPath),daemon=False)
        gpu_process = Process(target=runInference_Subprocess, args=(msg_queue,image_file,outname,modelPath))
        gpu_process.start()
        predictOutput = msg_queue.get()
        gpu_process.join()     
    elif (subprocessMethod == 'billiard'):
        print('starting a billiard subprocess')
        multiprocessing.set_start_method("spawn")
        msg_queue = Queue()
        gpu_process = billiard.Process(target=runInference_Subprocess, args=(msg_queue,image_file,outname,modelPath))
        gpu_process.start()
        predictOutput = msg_queue.get()
        gpu_process.join()     
    else:
        predictOutput = runInference(image_file,outname,modelPath)

    # write the output file to the named file
    with open(outname, 'w') as outfile:
        outfile.write(predictOutput)

    print('inferencing complete')

    # new output of segmentation statistics in a string
    statistics = c.generateStatsString(predictOutput)
    # generate unique names for multiple runs.  Add extension so it is easier to use

    statoutname = NamedTemporaryFile(delete=False).name+'_stats.json'
    open(statoutname,"w").write(statistics)

    # return the name of the output file and the stats
    return outname,statoutname


def runInference(image_file,outname,modelPath):
    # assume single user model, we aren't locking GPUs. use a single concurrency
    # option in girder_worker to force single user at a time
    c = extractPatch(modelPath)
    c.setArguments(image_file, outname,[5])
    predictOutput = c.parseMeta_and_pullTiles()
    return predictOutput


def runInference_Subprocess(msg_queue,image_file,outname,modelPath):
    # assume single user model, we aren't locking GPUs. use a single concurrency
    # option in girder_worker to force single user at a time
    c = extractPatch(modelPath)
    c.setArguments(image_file, outname,[5])
    predictOutput = c.parseMeta_and_pullTiles()
    msg_queue.put(predictOutput)



#---------------------------
# begin NIH AI code 
#---------------------------


class extractPatch:

    def __init__(self,modelPath):
        self.save_image_size = 500   # specify image size to be saved (note this is the same for all magnifications)
        self.pixel_overlap = 100       # specify the level of pixel overlap in your saved images
        self.limit_bounds = True     # this is weird, dont change it
        self.model_path = modelPath
 

    def setArguments(self,image_file,save_name,magnification=[5]):     
        self.image_file = image_file #'1044 - 2020-08-10 08.39.18.ndpi'#'1043 - 2020-08-10 08.48.55.ndpi'
        self.save_name = save_name #'CZ2'#'trythis'



    def parseMeta_and_pullTiles(self):
 
        #first load pytorch model
        learn = load_learner(self.model_path,cpu=False)

        # first grab data from digital header
        # print(os.path.join(self.file_location,self.image_file))
        oslide = openslide.OpenSlide(os.path.join(self.image_file))

        # this is physical microns per pixel
        acq_mag = 10.0/float(oslide.properties[openslide.PROPERTY_NAME_MPP_X])

        # this is nearest multiple of 20 for base layer
        base_mag = int(20 * round(float(acq_mag) / 20))

        # this is how much we need to resample our physical patches for uniformity across studies
        physSize = round(self.save_image_size*acq_mag/base_mag)

        # grab tiles accounting for the physical size we need to pull for standardized tile size across studies
        tiles = DeepZoomGenerator(oslide, tile_size=physSize-round(self.pixel_overlap*acq_mag/base_mag), overlap=round(self.pixel_overlap*acq_mag/base_mag/2), limit_bounds=self.limit_bounds)

        # calculate the effective magnification at each level of tiles, determined from base magnification
        tile_lvls = tuple(base_mag/(tiles._l_z_downsamples[i]*tiles._l0_l_downsamples[tiles._slide_from_dz_level[i]]) for i in range(0,tiles.level_count))

        # pull tiles from levels specified by self.mag_extract
        for lvl in self.mag_extract:
            if lvl in tile_lvls:
                # print(lvl)
                # pull tile info for level
                x_tiles, y_tiles = tiles.level_tiles[tile_lvls.index(lvl)]

                # note to self, we have to iterate b/c deepzoom does not allow casting all at once at list (??)
                polygons = []
                # xy_lim = self.get_box(path=self.xml_file)
                for y in range(0,y_tiles):
                    for x in range(0,x_tiles):

                        # grab tile coordinates
                        tile_coords = tiles.get_tile_coordinates(tile_lvls.index(lvl), (x, y))
                        save_coords = str(tile_coords[0][0]) + "-" + str(tile_coords[0][1]) + "_" + '%.0f'%(tiles._l0_l_downsamples[tile_coords[1]]*tile_coords[2][0]) + "-" + '%.0f'%(tiles._l0_l_downsamples[tile_coords[1]]*tile_coords[2][1])

                        tile_pull = tiles.get_tile(tile_lvls.index(lvl), (x, y))
                        ws = self.whitespace_check(im=tile_pull)
                        if ws < 0.95:
                            tile_pull = tile_pull.resize(size=(self.save_image_size, self.save_image_size),resample=PIL.Image.ANTIALIAS)
                            tile_pull = np.array(tile_pull)
                            inp, targ, pred, _ = learn.predict(tile_pull, with_input=True)
                            pred_arr = pred.cpu().detach().numpy()
                            img_arr = pred_arr.astype("bool")
                            pred_polys = self.tile_ROIS(imgname=save_coords,mask_arr=img_arr)
                            polygons += pred_polys

                predicton = self.slide_ROIS(polygons=polygons,mpp=float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]))

            else:
                print("WARNING: YOU ENTERED AN INCORRECT MAGNIFICATION LEVEL")

        return prediction

    def tile_ROIS(self,imgname,mask_arr):
        polygons = []
        nameparts = str.split(imgname, '_')
        pos = str.split(nameparts[0], '-')
        sz = str.split(nameparts[1], '-')
        radj = max([int(sz[0]), int(sz[1])]) / (self.save_image_size -1)
        start1 = int(pos[0])
        start2 = int(pos[1])
        c = morphology.remove_small_objects(mask_arr.astype(bool), 10, connectivity=2)
        c = morphology.binary_closing(c)
        c = morphology.remove_small_holes(c, 1000)
        contours, hier = cv2.findContours(c.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            cvals = contour.transpose(0, 2, 1)
            cvals = np.reshape(cvals, (cvals.shape[0], 2))
            cvals = cvals.astype('float64')
            for i in range(len(cvals)):
                cvals[i][0] = start1 + radj * (cvals[i][0])
                cvals[i][1] = start2 + radj * (cvals[i][1])
            try:
                poly = Polygon(cvals)
                if poly.length > 0:
                    polygons.append(Polygon(poly.exterior))
            except:
                pass

        return polygons

    def slide_ROIS(self,polygons,mpp):
        all_polys = unary_union(polygons)
        final_polys = []
        for poly in all_polys:
            #print(poly)
            if poly.type == 'Polygon':
                newpoly = Polygon(poly.exterior)
                if newpoly.area*mpp*mpp > 12000:
                    final_polys.append(newpoly)
            if poly.type == 'MultiPolygon':
                for roii in poly.geoms:
                    newpoly = Polygon(roii.exterior)
                    if newpoly.area*mpp*mpp > 12000:
                        final_polys.append(newpoly)
        final_shape = unary_union(final_polys)

        trythis = '['

        # write out the polygon coordinates discovered.  This causes a TypeError if 
        # len(final_shape) is undefined (when no polygons were discovered). so we added
        # an error catch here
        try:
            for i in range(0, len(final_shape)):
                trythis += json.dumps(
                    {"type": "Feature", "id": "PathAnnotationObject", "geometry": shapely.geometry.mapping(final_shape[i]),
                    "properties": {"classification": {"name": "Tumor", "colorRGB": -16711936}, "isLocked": False,
                                    "measurements": []}}, indent=4)
                if i < len(final_shape) - 1:
                    trythis += ','
            trythis += ']'
        except:
            # no polygons were found by the algorithm.  Return an empty list
            trythis = '[]'
        # return the output for postprocessing and download
        return trythis
      

    def whitespace_check(self,im):
        bw = im.convert('L')
        bw = np.array(bw)
        bw = bw.astype('float')
        bw=bw/255
        prop_ws = (bw > 0.8).sum()/(bw>0).sum()
        return prop_ws

        # ------ end NIH-AI Resource Team code --------------------------------


    # calculate the statistics for the image to be returned to the GUI for display

    def generateStatsString(predict_image):        
        statsDict = {'metric':1.0 }
        # convert dict to json string
        print('statsdict:',statsDict)
        statsString = json.dumps(statsDict)
        return statsString

