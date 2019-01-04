// import * as tf from '@tensorflow/tfjs';

// https://github.com/tensorflow/tfjs-examples/blob/master/simple-object-detection/train.js

let mymodel = null;
const CANVAS_SIZE = 224;  // Matches the input size of MobileNet.
const NUM_CLASSES = 2;
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
    //console.log("LOAD ", file);
    ImageTools.resize(file, {
       width: CANVAS_SIZE,
       height: CANVAS_SIZE
     }, function(blob, didItResize) {
        let image = new Image();
        //image.onload = function() {
          //const resized = imageToTensor(image)
        //};
        image.src = URL.createObjectURL(blob);
        image.title = file.name;
        image.className = "train-image";

        let div = document.createElement("div");
        div.className = "card";
        div.appendChild(image);
        preview.appendChild(div);
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

  let xs = new Array();
  let ys = new Array();
  console.log("MEM", tf.memory());
  for (var label = 0; label < 2; label++) {
    console.log("label ", label);
    let div = document.getElementById("preview" + (label + 1));
    let ch = div.childNodes;
    for (let i = 0; i < ch.length; i++) {
      if (i % 100 == 0) {
        console.log("i", i);
        console.log("MEM", tf.memory());
      }
      const div2 = ch[i];
      const img2 = div2.childNodes[0];
      // activation: [1, 7, 7, 256], float32
      // data() -> Float32Array(12544)
      const activation = tf.tidy( () => {
          const resized = imageToTensor(img2);
          const batched = resized.reshape([1, CANVAS_SIZE, CANVAS_SIZE, 3]);
          return truncatedMobileNet.predict(batched); // activation
        });
      console.log("A", activation);
      console.log("A", activation.shape);
      console.log("A", activation.data());      
      xs.push(activation);

      const y = tf.tidy(
          () => tf.oneHot(tf.tensor1d([label]).toInt(), NUM_CLASSES));
      ys.push(y);
    }
  }
  console.log("TICK/1");
  console.log("MEM", tf.memory());
  let bx = tf.concat(xs, 0);
  console.log("TICK/2");
  console.log("MEM", tf.memory());
  let by = tf.concat(ys, 0);
  console.log("MEM", tf.memory());
  console.log("Calling fit()");
  model2.fit(bx, by, {
             batchSize: 32,
          verbose: 0,
          epochs: 2,
          callbacks: {
     onEpochEnd: async (batch, logs) => {
          console.log("EPOCH LOSS: ", logs.loss.toFixed(5));
          console.log("MEM", tf.memory());
        },
     onTrainEnd: async (batch, logs) => {
          console.log("train end");
        }
      }
    });
}

function imageToTensor(image) {
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
  return resized;
}

async function init() {
  truncatedMobileNet = await loadTruncatedMobileNet();
  truncatedMobileNet.summary();
}

// Initialize the application.
init();

