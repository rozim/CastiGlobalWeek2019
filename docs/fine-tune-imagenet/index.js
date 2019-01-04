// import * as tf from '@tensorflow/tfjs';

// https://github.com/tensorflow/tfjs-examples/blob/master/simple-object-detection/train.js

let mymodel = null;
const CANVAS_SIZE = 224;  // Matches the input size of MobileNet.
const NUM_CLASSES = 2;
let truncatedMobileNet = null;
const normalizationOffset = tf.scalar(127.5);
const BATCH_SIZE = 16;
const EPOCHS = 10;
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

function shuffleArrays(a1, a2) {
  for (let i = a1.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a1[i], a1[j]] = [a1[j], a1[i]];
    [a2[i], a2[j]] = [a2[j], a2[i]];    
  }
}

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
  //model2.summary();

  let xs = new Array();
  let ys = new Array();
  console.log("MEM Loop begin", tf.memory());
  let mod = 1;
  for (var label = 0; label < 2; label++) {
    console.log("label ", label);
    let div = document.getElementById("preview" + (label + 1));
    let ch = div.childNodes;
    for (let i = 0; i < ch.length; i++) {
      if (i % mod == 0) {
        mod *= 2;
        console.log("i", i);
        console.log("Loop MEM", tf.memory());
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

      xs.push(activation.dataSync());
      const y = tf.tidy(
          () => tf.oneHot(tf.tensor1d([label]).toInt(), NUM_CLASSES));

      //Int32Array[2]
      ys.push(y.dataSync());
      tf.dispose([activation, y]);
    }
  }
  console.log("MEM Loop end", tf.memory());  

  console.log("Calling fit()");

  const stop = xs.length % BATCH_SIZE == 0 ? xs.length : xs.length - BATCH_SIZE;

  const p_epoch = document.getElementById("p_epoch");
  p_epoch.max = EPOCHS;
  const p_batch = document.getElementById("p_batch");
  p_batch.max = xs.length / BATCH_SIZE;
  for (var epoch = 0; epoch < EPOCHS; epoch++) {
    p_epoch.value = epoch;
    shuffleArrays(xs, ys);
    console.log("EPOCH START", epoch, tf.memory());
    for (var start = 0; start < stop; start += BATCH_SIZE) {
      p_batch.value = start / BATCH_SIZE;
      const bx = xs.slice(start, start + BATCH_SIZE);
      const by = ys.slice(start, start + BATCH_SIZE);
  
      const bxt = tf.tidy( () => tf.concat(bx).as4D(BATCH_SIZE, 7, 7, 256));
      const byt = tf.tidy( () => tf.concat(by).asType('float32').as2D(BATCH_SIZE, NUM_CLASSES));

      await model2.trainOnBatch(bxt, byt).then(loss =>
        {
          console.log("loss", loss);
          tf.dispose([bxt, byt]);
        });
    }
    console.log("EPOCH FINISHED", epoch, tf.memory());
  }
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
  //truncatedMobileNet.summary();
}

// Initialize the application.
init();

