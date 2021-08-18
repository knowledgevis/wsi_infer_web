# added for girder interaction as plugin arbor task
from girder_worker.app import app
from girder_worker.utils import girder_job
from tempfile import NamedTemporaryFile

# declared for subprocess to do GPU stuff.  Package 'billiard' comes with celery
# and is a workaround for subprocess limitations on 'daemonic' processes.

import billiard as multiprocessing
from billiard import Queue, Process
import json


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

    #print(" input image filename = {}".format(image_file))

    # setup the GPU environment for pytorch
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    DEVICE = 'cuda'

    print('perform forward inferencing')


    subprocess = False
    if (subprocess):
        # declare a subprocess that does the GPU allocation to keep the GPU memory from leaking
        msg_queue = Queue()
        gpu_process = Process(target=start_inference, args=(msg_queue,image_file))
        gpu_process.start()
        predict_image = msg_queue.get()
        gpu_process.join()     
    else:
        predict_image = start_inference_mainthread(image_file)
  
    predict_bgr = cv2.cvtColor(predict_image,cv2.COLOR_RGB2BGR)
    print('output conversion and inferencing complete')

    # generate unique names for multiple runs.  Add extension so it is easier to use
    outname = NamedTemporaryFile(delete=False).name+'.png'

    # write the output object using openCV  
    print('writing output')
    cv2.imwrite(outname,predict_bgr)
    print('writing completed')

    # new output of segmentation statistics in a string
    statistics = generateStatsString(predict_image)
    # generate unique names for multiple runs.  Add extension so it is easier to use

    statoutname = NamedTemporaryFile(delete=False).name+'.json'
    open(statoutname,"w").write(statistics)

    # return the name of the output file and the stats
    return outname,statoutname






def start_inference_mainthread(image_file):
    reset_seed(1)

    best_prec1_valid = 0.
    #torch.backends.cudnn.benchmark = True

    #saved_weights_list = sorted(glob.glob(WEIGHT_PATH + '*.tar'))
    saved_weights_list = [WEIGHT_PATH+'model_iou_0.4996_0.5897_epoch_45.pth.tar'] 
    print(saved_weights_list)

    print('about to instantiate model on GPU')
    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASS_VALUES),
        activation=ACTIVATION,
        aux_params=None,
    )

    print('model created')
    model = nn.DataParallel(model)
    print('data parallel done')
    model = model.cuda()
    print('moved to gpu.  now load pretrained weights')
    model = load_best_model(model, saved_weights_list[-1], best_prec1_valid)
    print('Loading model is finished!!!!!!!')

    # return image data so girder toplevel task can write it out
    predict_image = inference_image(model,image_file, BATCH_SIZE, len(CLASS_VALUES))

    # return the image to the main process
    return predict_image


# calculate the statistics for the image by converting to numpy and comparing masks against
# the tissue classes. create masks for each class and count the number of pixels

def generateStatsString(predict_image):
    # ERMS=red, ARMS=blue. Stroma=green, Necrosis = RG (yellow)
    img_arr = np.array(predict_image)
    # calculate total pixels = height*width
    total_pixels = img_arr.shape[0]*img_arr.shape[1]
    # count the pixels in the non-zero masks
    erms_count = np.count_nonzero((img_arr == [255, 0, 0]).all(axis = 2))
    stroma_count = np.count_nonzero((img_arr == [0, 255, 0]).all(axis = 2)) 
    arms_count = np.count_nonzero((img_arr == [0, 0, 255]).all(axis = 2)) 
    necrosis_count = np.count_nonzero((img_arr == [255, 255, 0]).all(axis = 2)) 
    print(f'erms {erms_count}, stroma {stroma_count}, arms {arms_count}, necrosis {necrosis_count}')
    erms_percent = erms_count / total_pixels * 100.0
    arms_percent = arms_count / total_pixels * 100.0
    necrosis_percent = necrosis_count / total_pixels * 100.0
    stroma_percent = stroma_count / total_pixels * 100.0
    # pack output values into a string returned as a file
    #statsString = 'ERMS:',erms_percent+'\n'+
    #              'ARMS:',arms_percent+'\n'+
    #              'stroma:',stroma_percent+'\n'+
    #              'necrosis:',necrosis_percent+'\n'
    statsDict = {'ERMS':erms_percent,
                 'ARMS':arms_percent, 
                 'stroma':stroma_percent, 
                 'necrosis':necrosis_percent }
    # convert dict to json string
    print('statsdict:',statsDict)
    statsString = json.dumps(statsDict)
    return statsString