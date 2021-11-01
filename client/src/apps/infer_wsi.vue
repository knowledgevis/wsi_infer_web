<template>
  <v-app>
    <v-layout class="transform-view" row fill-height>
      <v-navigation-drawer permanent fixed style="width: 400px; min-width: 400px;">
        <v-toolbar dark flat color="primary">
          <v-toolbar-title class="white--text">Preclinical Carcinoma Segmentation</v-toolbar-title>
        </v-toolbar>
        <v-spacer/>
        <v-container fluid>

          <v-flex xs12>
            <v-btn
             outline
             block
              @click="loadSampleImageFile"
            >
            Use a Provided Sample Image
            </v-btn>
          </v-flex>

          <v-flex xs12>
            <v-btn class="text-none" outline block @click='$refs.imageFile.click()'>{{ fastaFileName || 'UPLOAD Whole Slide Image' }}</v-btn>
            <input
              type="file"
              style="display: none"
              ref="imageFile"
              @change="uploadImageFile($event.target.files[0])"
            >
          </v-flex>
          <v-flex xs12>
            <v-btn
              block
              :class="{ primary: readyToRun }"
              :flat="readyToRun"
              :outline="!readyToRun"
              :disabled="!readyToRun"
              @click="run"
            >
              Go
            </v-btn>
          </v-flex>
          <v-flex xs12>
            <v-btn
              block
              :class="{ primary: readyToDownload }"
              :flat="readyToDownload"
              :outline="!readyToDownload"
              :disabled="!readyToDownload"
              @click="downloadResults"
            >
              Download Results 
            </v-btn>
          </v-flex>
          <v-flex xs12>
            <v-btn
              block
              :class="{ primary: readyToDownload }"
              :flat="readyToDownload"
              :outline="!readyToDownload"
              :disabled="!readyToDownload"
              @click="reset"
            >
              Prepare For Another Image 
            </v-btn>
          </v-flex>
        </v-container>
      </v-navigation-drawer>
      <v-layout column justify-start fill-height style="margin-left: 400px">
          <v-card class="ma-4">
            <v-card-text>
              <b>This application detects and identifies regions of adenocarcinoma from a whole slide image by executing a 
                neural network pre-trained by the NIH AI Resource (AIR) team. Uploaded images can be in any of 
                the standard whole slide image formats.
              <br><br>
              After selecting an image for upload, please be patient during the upload process, as WSIs can take 
              a long time to transfer. Once a thumbnail of the input mage is displayed below, please click the "Go" 
              button to begin execution. After analysis begins, please again be patient, as the neural network-based 
              analysis will take several minutes, depending on the size of the input image being provided. When the 
              analysis is complete, the resulting segmentation will be displayed below and will be available for downloading, 
              using the download button. If you would like to segment additional images, please just click "Prepare for 
              Another Image" in between each segmentation operation. This tells the system to reset and prepare to run again.
                <br><br>
              We are delighted that you are trying our early release system for adenocarcinoma analysis. Thank you.  
              If you have any questions while using our system, please email Dr. Stephanie Harmon at stephanie.harmon@nih.gov.  
              </b>
            </v-card-text>
          </v-card>

           <div v-if="uploadIsHappening" xs12 class="text-xs-center mb-4 ml-4 mr-4">
              Image Upload in process...
              <v-progress-linear indeterminate=True></v-progress-linear>
          </div>

          <div v-if="thumbnailComplete">
            <div  xs12 class="text-xs-center mb-4 ml-4 mr-4">
              <v-card class="mb-4 ml-4 mr-4">
                <v-card-text>Uploaded Image</v-card-text>
                  <img :src="inputImageUrl" style="display: block; margin: auto"> 
                </v-card>
            </div>
          </div>

        <v-card v-if="running && job.status==0" xs12 class="text-xs-center mb-4 ml-4 mr-4">
            Another user is currently using the system.  Please wait.  Your inferencing job should begin automatically when the previous job completes. 
            <v-progress-linear indeterminate=True></v-progress-linear>
        </v-card>

        <v-card v-if="running" xs12 class="text-xs-center mb-4 ml-4 mr-4">
            Running Neural Network... please wait for the output to show below. This will take several minutes.
          <v-progress-linear indeterminate=True></v-progress-linear>
        </v-card>

        <div v-if="runCompleted" xs12 class="text-xs-center mb-4 ml-4 mr-4">
          Job Complete  ... 
        </div>

        <div v-if="thumbnailComplete">

          <v-card  align="center" justify="center" class="mt-20 mb-4 ml-4 mr-4">
              <div id="visM" ref="visModel" class="mt-20 mb-4 ml-4 mr-4"></div>
          </v-card>

          <v-card v-if="table.length > 0" class="mt-8 mb-4 ml-4 mr-4">
            <v-card-text>Image Statistics</v-card-text>
            <json-data-table :data="table" />
          </v-card>

        </div>



      </v-layout>
    </v-layout>
  </v-app>
</template>

<script>

import { utils } from '@girder/components/src';
import { csvParse } from 'd3-dsv';
import scratchFolder from '../scratchFolder';
import pollUntilJobComplete from '../pollUntilJobComplete';
import optionsToParameters from '../optionsToParameters';
import JsonDataTable from '../components/JsonDataTable';
import OpenSeadragon from 'openseadragon';
import vegaEmbed from 'vega-embed';


export default {
  name: 'infer_rhabdo',
  inject: ['girderRest'],
  components: {
    JsonDataTable,
  },
  data: () => ({
    imageFile: {},
    imageFileName: '',
    imagePointer: '',
    imageBlob: [],
    uploadedImageUrl: '',
    job: { status: 0 },
    result: { status: 0 },
    readyToDisplayInput: false,
    running: false,
    thumbnail: [],
    thumbnailComplete: false,
    result: [],
    table: [],
    stats: [],
    data: [],
    resultColumns: [],
    resultString:  '',
    runCompleted: false,
    uploadInProgress: false,
    inputImageUrl: '',
    outputImageUrl: '',
    inputDisplayed:  false,
    outputDisplayed:  false,
    osd_viewer: [],
    imageStats: {},
  }),
  asyncComputed: {
    scratchFolder() {
      return scratchFolder(this.girderRest);
    },
  },
  computed: {
    readyToRun() {
      return !!this.imageFileName; 
    },
    readyToDownload() {
      return (this.runCompleted)
    },
    uploadIsHappening() {
      return (this.uploadInProgress)
    },
  },

  methods: {

    // method here to create and display a thumbnail of an arbitrarily large whole slilde image.
    // This code is re-executed for each UI change, so the code is gated to only run once 

    async renderInputImage() {
       if (this.inputDisplayed == false) {

        // create a spot in Girder for the output of the REST call to be placed
          const outputItem = (await this.girderRest.post(
            `item?folderId=${this.scratchFolder._id}&name=thumbnail`,
          )).data

        // build the params to be passed into the REST call
        const params = optionsToParameters({
          imageId: this.imageFile._id,
          outputId: outputItem._id,
        });
        // start the job by passing parameters to the REST call
        this.job = (await this.girderRest.post(
          `arbor_nova/wsi_thumbnail?${params}`,
        )).data;

          // wait for the job to finish
          await pollUntilJobComplete(this.girderRest, this.job, job => this.job = job);

          if (this.job.status === 3) {
            this.running = false;
            // pull the URL of the output from girder when processing is completed. This is used
            // as input to an image on the web interface
            this.thumbnail = (await this.girderRest.get(`item/${outputItem._id}/download`,{responseType:'blob'})).data;
            // set this variable to display the resulting output image on the webpage 
            this.inputImageUrl = window.URL.createObjectURL(this.thumbnail);

          }

          console.log('render input finished')
	        this.inputDisplayed = true
          this.thumbnailComplete = true;
          this.uploadInProgress = false;
	     }
    },



    async run() {
      this.running = true;
      this.errorLog = null;

      // create a spot in Girder for the output of the REST call to be placed
      const outputItem = (await this.girderRest.post(
        `item?folderId=${this.scratchFolder._id}&name=result`,
      )).data

      // create a spot in Girder for the output of the REST call to be placed
      const statsItem = (await this.girderRest.post(
        `item?folderId=${this.scratchFolder._id}&name=stats`,
      )).data

      // build the params to be passed into the REST call
      const params = optionsToParameters({
        imageFileName: this.imageFileName,
        imageId: this.imageFile._id,
        outputId: outputItem._id,
        statsId: statsItem._id
      });
      console.log('params to inference',params)
 
      // start the job by passing parameters to the REST call
      console.log('starting backend inference')
      this.result = (await this.girderRest.post(
        `inference?${params}`,
      )).data;
      console.log('inference result',this.result)
  
      if (this.result.status === "success") {
        this.running = false;
        this.runCompleted = true;

        // copy this data to a state variable for rendering in a table. convert from a string
        this.data = JSON.parse(this.result.result)
        console.log ('json to plot:',this.data)
        this.stats = JSON.parse(this.result.stats)

        // build the spec here.  Inside the method means that the data item will be available. 
        let titleString = 'Number of adenocarcinoma contours predicted'

        var vegaLiteSpec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v4.json",
            "description": "A simple bar chart with embedded data.",
             title: titleString,
              "height": 200,
              "width": 600,
              "autosize": {
                "type": "fit",
                "contains": "padding"
              },
            "data": {
              "values": [
                {"Class": "Regions","count": this.stats.numberOfRegions}, 
             
              ]
            },
           "layer": [{
              "mark": "bar"
            }, {
              "mark": {
                "type": "text",
                "align": "center",
                "baseline": "bottom",
                "fontSize": 13,
                "dx": 25
              },
              "encoding": {
                "text": {"field": "count", "type": "quantitative"}
              }
            }],
            "encoding": {
              "y": {"field": "Class", "type": "ordinal", "title":""},
              "x": {"field": "count", "type": "quantitative", "title":"Count of separate cancerous regions"},
              "color": {
                  "field": "Class",
                  "type":"nominal",
                  "scale": {"domain":["Regions"],"range": ["blue"]}
                  }
            }
          };

          // render the chart with options to save as PNG or SVG, but other options turned off
          vegaEmbed(this.$refs.visModel,vegaLiteSpec,
                 {padding: 10, actions: {export: true, source: false, editor: false, compiled: false}});


      }
    },


    async uploadImageFile(file) {
      if (file) {
        this.runCompleted = false;
        this.imageFileName = file.name;
        this.uploadInProgress = true;
        const uploader = new utils.Upload(file, {$rest: this.girderRest, parent: this.scratchFolder});
        this.imageFile = await uploader.start();
        // display the uploaded image on the webpage
        console.log('image uploader result',this.imageFile)
	      console.log('displaying input image...');
        //this.imageBlob = (await this.girderRest.get(`file/${this.imageFile._id}/download`,{responseType:'blob'})).data;
        //this.uploadedImageUrl = window.URL.createObjectURL(this.imageBlob);
	      //console.log('createObjURL returned: ',this.uploadedImageUrl);
        this.readyToDisplayInput = true;
        this.renderInputImage();
      }
    },

  async loadSampleImageFile() {
    console.log('load sample image')
    this.runCompleted = false;
    this.uploadInProgress = true;
    this.imageFileName = 'UniqueSampleImageNameAIR'
    const params = optionsToParameters({
          q: this.imageFileName,
          types: JSON.stringify(["file"])
        });
    // find the sample image already uploaded in Girder
    this.fileId = (await this.girderRest.get(
      `resource/search?${params}`,
    )).data["file"][0];

    console.log('displaying sample input stored at girder ID:',this.fileId);
    this.imageFile = this.fileId
    this.inputDisplayed == false;
    this.readyToDisplayInput = true;
    this.renderInputImage();
    },

    // when the user clicks download, download this through the browser.  Name it the same as the uploaded file with
    // _prediction.json appended to the name
    downloadResults() {
        var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(this.data));
        var dlAnchorElem = document.createElement('a');
        dlAnchorElem.setAttribute("href",     dataStr     );
        var outfilename = "adenocarcinoma_prediction.json"
        dlAnchorElem.setAttribute("download", outfilename);
        document.body.appendChild(dlAnchorElem);
        dlAnchorElem.click();
        document.body.removeChild(dlAnchorElem);
    },



    // reload the page to allow the user to process another image.
    // this clears all state and image displays. The scroll command
    // resets the browser to the top of the page. 
    reset() {
      window.location.reload(true);
      window.scrollTo(0,0);
      this.thumbnailComplete=false;
      this.runCompleted=false;
    },
  }
}
</script>
