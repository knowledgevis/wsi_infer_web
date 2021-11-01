from girder import plugin
from girder.plugin import GirderPlugin
from tempfile import NamedTemporaryFile

from girder.api.rest import Resource
from girder.api.describe import Description, describeRoute, RestException
from girder.api import access
from girder.models.file import File

import pymongo
from pymongo import MongoClient
import girder_client
import json
import logging
import arrow
import datetime

#import requests
import string
import sys

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


# email modules
#import smtplib
#from email.mime.text import MIMEText

# globals
globals = {}
globals['mongo-database'] = 'NIH_ai_infer'
globals['mongo-host'] = 'localhost'
globals['mongo-log-collection'] = 'logging'
globals['mongo-port'] = 27017
globals['timezone'] = 'US/Eastern'
globals['modelPath'] = './models/deeplabv3_resnet50_10ep_lr1e4_nonorm.pkl'
globals['girderUser'] = 'anonymous'
globals['girderPassword'] = 'letmein'


client = None
log_collection = None

# declare the handlers for get, put, post, delete
class ImageInference_API(Resource):
    def __init__(self):
        super(ImageInference_API, self).__init__()
        self.resourceName = 'inference'

        #the POST is used to retrieve 
        self.route('GET', (), self.parseGetCommands)
        self.route('PUT', (), self.parsePutCommands)
        self.route('POST', (), self.performInference)
        #self.route('DELETE', (':id',), self.deleteRemoteIntakeRecords)

  

    # define a handler for the GET option that looks for a standard format command/data object
    # and dispatches to the correct handling routine.  This is organized where there is a function for 
    # each command.  This dispatcher just checks that the arguments are valid and calls the correct handler
 
    @access.public
    @describeRoute( 
        Description('dispatcher for GET API calls')
        .param('command', 'the api command string.', required=True,paramType='query')
        .param('data', 'the  command data as a JSON object.', required=False,paramType='query')
        .errorResponse())

    def parseGetCommands(self,params):
        # made the data param optonal because some commands might not need it?  Maybe this is inviting more 
        # parsing, but the testing could 
        self.requireParams(('command'), params)
        print('received GET command with params:',params)

        # check that the URL has the proper arguments (a command and a data argument) 
        try:
            commandName = params['command']
            print('infer: received command:', commandName)
        except ValueError:
            raise RestException('Invalid JSON command name passed in request body.')
        # dispatch successful commands
        if params['command'] == 'get_log':
            return self.getLogRecords()       
        elif params['command'] == 'get_stats':
            return self.getStats(params)     
        else:
            print('infer: incorrect command for GET dispatch detected:', params)
            response = {'status':'failure','data':params}
            return response



    @access.public
    @describeRoute( 
        Description('dispatcher for PUT API calls')
        .param('command', 'the api command string.', required=True,paramType='query')
        .param('data', 'the  command data as a JSON object.', required=False,paramType='query')
        .errorResponse())

    def parsePutCommands(self,params):
        print('received a PUT command')
        pass


    # configure the logger and open a database connection
    def setupLogging(self):
        global log_collection
        global client

        # setup database-based log
        client = MongoClient(globals['mongo-host'] ,globals['mongo-port'])
        db = client[globals['mongo-database']] 
        log_collection = db[globals['mongo-log-collection']]

    # do any cleanup 
    def closeLogChannels(self):
        global client
        client.close()

    def logActivityDetails(self,message,level='Info'):
        global log_collection
        # write database  with timestamp
        utc = arrow.utcnow()
        localtime = utc.to(globals['timezone'] )
        timestring = localtime.format('YYYY-MM-DD HH:mm:ss ZZ')
        logstring = timestring+' '+message
        log_collection.insert({'timestamp': logstring, 'message': message,
            'year':localtime.year,
            'month':localtime.month,
            'day':localtime.day,
            'hour':localtime.hour,
            'weekday':localtime.weekday()}
            )


    # routine that writes log message and writes them to the database 
    def logActivity(self,message,loglevel='Info'):
        self.setupLogging()
        self.logActivityDetails(message,loglevel)
        self.closeLogChannels()


    # return the records of system use by reading the log in the mongoDB instance    
    def getLogRecords(self):
        client = MongoClient(globals['mongo-host'] ,globals['mongo-port'])
        db = client[globals['mongo-database']]
        form_collection = db[globals['mongo-log-collection']]
        # put the entire record into the mongo database
        query = {}
        records = form_collection.find(query,{})
        response = {}
        # copy the records into a list and return the list
        logrecords = []
        for x in records:
            logrecords.append(x)
            print('log list: returning ',x)
        response['result'] = logrecords
        response['status'] = 'success'
        client.close()
        return response

    # dummy placeholder
    def getStats(self):
        pass


    # ************************* image inferencing entry point *****

    # define handle dispatcher for POST calls. 
    @access.public
    @describeRoute( 
        Description('dispatcher for POST API calls')
        .param('data', 'the  command data as a JSON object.', required=False,paramType='query')
        .errorResponse())

    def performInference(self,params):   
        print('POST params:',params)
       
        try:
            print('params:',params)
            imageId = params['imageId']
            outname = params['outputId']
            print('imageId',imageId)
        except:
            print('could not read image file variable')

        # setup the GPU environment for pytorch
        #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        DEVICE = torch.device(f"cuda:0")

        print('perform forward inferencing')

        # generate a spot in Girder for the output data file (a geoJSON file)
        # We generate unique names for multiple runs.
        try:  
            outname = 'tumor_predict_'+imageId+'.json'
        except:
            outname = 'tumor_predict_'+NamedTemporaryFile(delete=False).name+'.json'
        print('outname:',outname)

        print('invoke deep learning script')
        c = extractPatch(globals['modelPath'])
        print('extract patch')
        c.setArguments(imageId, outname,[5])
        print('set arguments')
        predictOutput = c.parseMeta_and_pullTiles(imageId)
        predictAsTopo = c.GeoToTopoJson(predictOutput)

        print('parseMeta')
        statistics = c.generateStatsString(predictOutput)
        print('stats')

        # write the output to the named file
        #with open(outname, 'w') as outfile:
        #    outfile.write(predictOutput)

        print('inferencing complete')

        # return the output JSON and the stats
        response = {'status':'success','stats':statistics, 'result':predictOutput,'topojson':predictAsTopo,'outname':outname}
        return response


#---------------------------
# begin NIH AI code 
#---------------------------


class extractPatch:

    def __init__(self,modelPath):
        self.save_image_size = 500   # specify image size to be saved (note this is the same for all magnifications)
        self.pixel_overlap = 100       # specify the level of pixel overlap in your saved images
        self.limit_bounds = True     # this is weird, dont change it
        self.model_path = modelPath
        self.image_file = ''
        self.imageId = None

    def setArguments(self,imageId,save_name,magnification=[5]):     
        self.imageId = imageId #'1044 - 2020-08-10 08.39.18.ndpi'#'1043 - 2020-08-10 08.48.55.ndpi'
        self.save_name = save_name #'CZ2'#'trythis'
        self.mag_extract = magnification



    def parseMeta_and_pullTiles(self, imageId):
 
        #first load pytorch model
        print('loading learner')
        learn = load_learner(self.model_path,cpu=False)
        print('learner loaded')

        # fetch the actual image data from the uploaded record
        print('finding image in girder backend')
        gc = girder_client.GirderClient(apiUrl='http://localhost:8080/girder/api/v1')
        login = gc.authenticate(globals['girderUser'],globals['girderPassword'])
        print('logged into girder successfully.')
        print('trying to local filename of file',imageId)
        fileRec = gc.getFile(imageId)
        #print('found file',fileRec['_id'])
        print('file record',fileRec)
        # hard to find the file on the disk, so download again.  Inefficient, but it works
        print('downloading file')
        gc.downloadFile(fileRec['_id'],'imageFile')
        self.image_file = 'imageFile'
        print('setting infile name and downloaded it')

        # get local filename from girder
        #file = File().load(imageId, user=globals['girderUser'])
        #self.image_file = File().getLocalFilePath(file)

        print('image is at:',self.image_file)

        # first grab data from digital header
        # print(os.path.join(self.file_location,self.image_file))
        oslide = openslide.OpenSlide(os.path.join(self.image_file))
        print('openslide was able to read image')

        # this is physical microns per pixel
        acq_mag = 10.0/float(oslide.properties[openslide.PROPERTY_NAME_MPP_X])
        print('microns per pixel',acq_mag)

        # this is nearest multiple of 20 for base layer
        base_mag = int(20 * round(float(acq_mag) / 20))
        print('base mag')

        # this is how much we need to resample our physical patches for uniformity across studies
        physSize = round(self.save_image_size*acq_mag/base_mag)
        print('physSize')

        # grab tiles accounting for the physical size we need to pull for standardized tile size across studies
        tiles = DeepZoomGenerator(oslide, tile_size=physSize-round(self.pixel_overlap*acq_mag/base_mag), overlap=round(self.pixel_overlap*acq_mag/base_mag/2), limit_bounds=self.limit_bounds)
        print('tiles')

        # calculate the effective magnification at each level of tiles, determined from base magnification
        tile_lvls = tuple(base_mag/(tiles._l_z_downsamples[i]*tiles._l0_l_downsamples[tiles._slide_from_dz_level[i]]) for i in range(0,tiles.level_count))
        print('lvls',tile_lvls)

        # pull tiles from levels specified by self.mag_extract
        for lvl in self.mag_extract:
            if lvl in tile_lvls:
                print(lvl)
                # pull tile info for level
                x_tiles, y_tiles = tiles.level_tiles[tile_lvls.index(lvl)]

                # note to self, we have to iterate b/c deepzoom does not allow casting all at once at list (??)
                polygons = []
                # xy_lim = self.get_box(path=self.xml_file)
                print('x,y tiles',x_tiles,y_tiles)
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

                prediction = self.slide_ROIS(polygons=polygons,mpp=float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]))

            else:
                print("WARNING: YOU ENTERED AN INCORRECT MAGNIFICATION LEVEL")
        print('parseMeta and pullTiles finished')
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

        # if there were no polygons found, this code generates a runtime exception, so 
        # catch the exception and return an empty list
        try:
            for i in range(0, len(final_shape)):
                trythis += json.dumps(
                    {"type": "Feature", "id": "PathAnnotationObject", "geometry": shapely.geometry.mapping(final_shape[i]),
                    "properties": {"classification": {"name": "Tumor", "colorRGB": -16711936}, "isLocked": False,
                                    "measurements": []}}, indent=4)
                if i < len(final_shape) - 1:
                    trythis += ','
            trythis += ']'
        except TypeError:
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

    # calculate the output data and statistics for the prediction to be returned to the GUI for display

    def GeoToTopoJson(self, geostring):
        # see http://github.com/topojson/topojson-specification
        geojson = json.loads(geostring)
        transform = {"scale":[1,1],"translate":[0,0]}
        topo = {'type': 'Topology', 'transform':transform}
        exampleObj = {"type": "GeometryCollection"}
        exampleObj['geometries'] = geojson
        topo['objects'] = exampleObj
        return json.dumps(topo)

    # count the number of polygons so we can print a graph of the number of detected regions
    def generateStatsString(self,geostring):         
        geojson = json.loads(geostring)
        numPolys = len(geojson)    
        statsDict = {'numberOfRegions':numPolys }
        # convert dict to json string
        print('statsdict:',statsDict)
        statsString = json.dumps(statsDict)
        return statsString


        # ------ end NIH-AI Resource Team code --------------------------------



#---------------------------



# for Girder 3.0: we need to declare the plugin using a 
# different class.  The load function is required to initialize 
# the plugin code and specify the route (.inference) where the new 
# functions will be accessed

class GirderPlugin(plugin.GirderPlugin):
    DISPLAY_NAME = 'inference'

    def load(self,info):
        info['apiRoot'].inference = ImageInference_API()


