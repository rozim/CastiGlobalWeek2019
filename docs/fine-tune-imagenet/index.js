// import * as tf from '@tensorflow/tfjs';
// import {ControllerDataset} from './controller_dataset';

// https://github.com/tensorflow/tfjs-examples/blob/master/simple-object-detection/train.js

let mymodel = null;
const CANVAS_SIZE = 224;  // Matches the input size of MobileNet.
const NUM_CLASSES = 2;
const controllerDataset = new ControllerDataset(NUM_CLASSES);
let truncatedMobileNet = null;
const normalizationOffset = tf.scalar(127.5);

// Name prefixes of layers that will be unfrozen during fine-tuning.
// const topLayerGroupNames = ['conv_pw_11'];

// Name of the layer that will become the top layer of the truncated base.
// const topLayerName =
//`${topLayerGroupNames[topLayerGroupNames.length - 1]}_relu`;

/*
console.log("Loading");
mobilenet.load().then(model => {
    mymodel = model;
    console.log("Loaded");
  });
*/

document.getElementById("select1").onchange = function(evt) {
  addUploadedImages(this.files, document.querySelector("#preview1", 0));
};

document.getElementById("select2").onchange = function(evt) {
  addUploadedImages(this.files, document.querySelector("#preview2", 1));
};

function addUploadedImages(files, preview, classNumber) {
  if (files) {
    [].forEach.call(files, readAndPreview);
  }

  function readAndPreview(file) {
    // console.log("LOAD ", file);
    ImageTools.resize(file, {
       width: CANVAS_SIZE,
       height: CANVAS_SIZE
     }, function(blob, didItResize) {
        let image = new Image();
        image.onload = function() {

          // Line 100+
          // https://github.com/tensorflow/tfjs-models/blob/master/mobilenet/src/index.ts
          const t = tf.fromPixels(image);
          // Normalize the image from [0, 255] to [-1, 1].
          const normalized = t.toFloat()
            .sub(normalizationOffset)
            .div(normalizationOffset); // as tf.Tensor3D;

          let resized = normalized;
          if (t.shape[0] !== CANVAS_SIZE || t.shape[1] !== CANVAS_SIZE) {
            const alignCorners = true;
            resized = tf.image.resizeBilinear(
                normalized, [CANVAS_SIZE, CANVAS_SIZE], alignCorners);
          }

          //// Reshape to a single-element batch so we can pass it to predict.
          //const batched = resized.reshape([1, CANVAS_SIZE, CANVAS_SIZE, 3]);
          controllerDataset.addExample(resized, classNumber);
        };        
        image.src = URL.createObjectURL(blob);
        image.title = file.name;
        image.className = "train-image";

        let div = document.createElement("div");
        div.className = "card";
        div.appendChild(image);
        preview.appendChild(div);


        return;
          
          


      });
  }
}

function b2() {
  console.log("b2");
  let div = document.getElementById("preview2");
  let ch = div.childNodes;
  for (let i = 0; i < ch.length; i++) {
    let div2 = ch[i];
    let img2 = div2.childNodes[0];
    mymodel.classify(img2, /*topk*/1).then(predictions => {
        console.log("pred: ", predictions);
      });
  }
}

async function loadTruncatedMobileNet() {
  const mobilenet = await tf.loadModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // Return a model that outputs an internal activation.
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function loadit2() {


  let model2 = tf.sequential({
   layers: [
       // Flattens the input to a vector so we can use it in a dense layer. While
       // technically a layer, this only performs a reshape (and has no training
       // parameters).
       tf.layers.flatten({
        inputShape: truncatedMobileNet.outputs[0].shape.slice(1)
         }),
       // Layer 1.
       tf.layers.dense({
        units: 10, // TBD: 
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
         }),
       // Layer 2. The number of units of the last layer should correspond
       // to the number of classes we want to predict.
       tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
         })
           ]
    });

  const optimizer = tf.train.adam(1e-3);
  model2.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
  model2.summary();
  console.log("XXX");
  truncatedMobileNet.summary();

  model2.fit(controllerDataset.xs, controllerDataset.ys, {
      batchSize,
          epochs: 2,
          callbacks: {
     onBatchEnd: async (batch, logs) => {
          console.log("LOSS: ", logs.loss.toFixed(5));
        }
      }
    });
}

async function init() {
  truncatedMobileNet = await loadTruncatedMobileNet();
  truncatedMobileNet.summary();
}

// Initialize the application.
init();

