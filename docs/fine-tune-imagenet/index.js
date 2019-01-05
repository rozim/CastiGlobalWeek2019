// import * as tf from '@tensorflow/tfjs';

// https://github.com/tensorflow/tfjs-examples/blob/master/simple-object-detection/train.js

let mymodel = null;
const CANVAS_SIZE = 224;  // Matches the input size of MobileNet.
const NUM_CLASSES = 2;
let truncatedMobileNet = null;
const normalizationOffset = tf.scalar(127.5);
const BATCH_SIZE = 8;
const EPOCHS = 100;
const PICK = 0.85;
let global_model2;

const LEARNING_RATE = 0.0001;

const HIDE = "&#9654;";
const SHOW = "&#9660;";

function shuffleArrays(a1, a2) {
  for (let i = a1.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a1[i], a1[j]] = [a1[j], a1[i]];
    [a2[i], a2[j]] = [a2[j], a2[i]];    
  }
}

document.getElementById("select1").onchange = function(evt) {
  addUploadedImages(this.files, document.querySelector("#preview1"), 0);
};

document.getElementById("select2").onchange = function(evt) {
  addUploadedImages(this.files, document.querySelector("#preview2"), 1);
};

function addUploadedImages(files, preview, classNumber) {
  console.log(files, preview, classNumber);
  /*
  if (files) {
    [].forEach.call(files, readAndPreview);
  }
  */

  for (let i = 0; i < files.length; i++) {
    readAndPreview(files[i]);
  }

  function readAndPreview(file) {
    //console.log("LOAD ", file);
    ImageTools.resize(file, {
       width: CANVAS_SIZE,
       height: CANVAS_SIZE
     }, function(blob, didItResize) {
        let image = new Image();
        image.src = URL.createObjectURL(blob);
        image.title = file.name;
        image.className = "train-image";
        image.width = CANVAS_SIZE;
        image.height = CANVAS_SIZE;        
        let div = document.createElement("div");
        div.className = "card";
        div.appendChild(image);
        preview.appendChild(div);

        /*
        const onload = (my_image) => (event) => {
          const resized = imageToTensor(my_image); 
          const batched = resized.reshape([1, CANVAS_SIZE, CANVAS_SIZE, 3]);
          my_image.activation = truncatedMobileNet.predict(batched); // activation

        };

        image.onload = onload(image);
        */
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
        units: 100, // TBD:
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true,
        name: 'my_dense',
         }),
       
       // Layer 2. The number of units of the last layer should correspond
       // to the number of classes we want to predict.
       tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax',
        name: 'my_softmax'
         })
           ]
    });
  const surface = { tab: 'Model Summary', name: 'MyModel' };
  tfvis.show.modelSummary(surface, model2);
  
  global_model2 = model2;

  let s = { tab: 'Details', name: 'my_softmax'};
  tfvis.show.layer(s, model2.getLayer('my_softmax'));
  s = { tab: 'Details', name: 'my_dense'};
  tfvis.show.layer(s, model2.getLayer('my_dense'));  

  const optimizer = tf.train.adam(LEARNING_RATE);
  model2.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

  let xs_all = new Array();
  let ys_all = new Array();
  console.log("MEM Loop begin", tf.memory());
  let mod = 16;

  let num_images = 0;
  for (var label = 0; label < NUM_CLASSES; label++) {
    num_images += document.getElementById("preview" + (label + 1)).childNodes.length;
  }

  let cur_image = 0;
  for (var label = 0; label < NUM_CLASSES; label++) {
    console.log("label ", label);
    let div = document.getElementById("preview" + (label + 1));
    let ch = div.childNodes;
    for (let i = 0; i < ch.length; i++) {
      cur_image += 1;
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

      xs_all.push(activation.dataSync());
      const y = tf.tidy(
          () => tf.oneHot(tf.tensor1d([label]).toInt(), NUM_CLASSES));


      //Int32Array[2]
      ys_all.push(y.dataSync());
      tf.dispose([activation, y]);
    }
  }
  console.log("MEM Loop end", tf.memory());

  shuffleArrays(xs_all, ys_all);

  const pick = Math.ceil(PICK * xs_all.length);
  const xs_train = xs_all.slice(0, pick);
  const ys_train = ys_all.slice(0, pick);
  const xs_validation = xs_all.slice(pick, xs_all.length);
  const ys_validation = ys_all.slice(pick, ys_all.length);
  console.log("PICK", pick, xs_train.length, xs_validation.length, xs_all.length);  
  xs_all = undefined;
  ys_all = undefined;

  console.log("Calling fit()");

  const stop = xs_train.length % BATCH_SIZE == 0 ? xs_train.length : xs_train.length - BATCH_SIZE;

  const p_epoch = document.getElementById("p_epoch");
  p_epoch.max = EPOCHS;
  const p_batch = document.getElementById("p_batch");
  p_batch.max = xs_train.length / BATCH_SIZE;
  const losses = new Array();
  const batch_losses = new Array();
  const times = new Array();
  const accuracys = new Array();
  let batch_number = 0;
  for (var epoch = 0; epoch < EPOCHS; epoch++) {
    p_epoch.value = epoch + 1;
    shuffleArrays(xs_train, ys_train);
    let sum_loss = 0.0;
    const t1 = new Date().getTime();
    for (var start = 0; start < stop; start += BATCH_SIZE) {
      p_batch.value = start / BATCH_SIZE;
      const bx = xs_train.slice(start, start + BATCH_SIZE);
      const by = ys_train.slice(start, start + BATCH_SIZE);
  
      const bxt = tf.tidy( () => tf.concat(bx).as4D(BATCH_SIZE, 7, 7, 256));
      const byt = tf.tidy( () => tf.concat(by).asType('float32').as2D(BATCH_SIZE, NUM_CLASSES));


      await model2.trainOnBatch(bxt, byt).then(loss =>
        {
          sum_loss += loss;
          tf.dispose([bxt, byt]);
          batch_losses.push({x: batch_number, y: loss});
          batch_number += 1;

          const series = ['Batch Loss'];
          const data = { values: [batch_losses], series }
          const surface = tfvis.visor().surface({ tab: 'Training',
                                                  name: 'Batch Loss',
                                                  styles: { height: 200, maxHeight: 200 },
            });
          tfvis.render.linechart(data, surface);          
        });
    } // end of one epoch of training loop

    const t2 = new Date().getTime();
    times.push({x: epoch, y: (t2 - t1) / 1000.0});
    losses.push({x: epoch, y: sum_loss});

    // begin validation

    const vstop = xs_validation.length % BATCH_SIZE == 0 ? xs_validation.length : xs_validation.length - BATCH_SIZE;
    console.log("xxx_val", xs_validation.length, vstop);
    let num_right = 0;
    let num_wrong = 0;
    let num = 0;
    for (var start = 0; start < vstop; start += BATCH_SIZE) {
      tf.tidy( () => {
          const bx = xs_validation.slice(start, start + BATCH_SIZE);
          const by = ys_validation.slice(start, start + BATCH_SIZE);
          const bxt = tf.concat(bx).as4D(BATCH_SIZE, 7, 7, 256);
          const byt = tf.concat(by).asType('float32').as2D(BATCH_SIZE, NUM_CLASSES);

          const softmaxes = model2.predictOnBatch(bxt);
          const pred = tf.argMax(softmaxes, 1);
          const goal = tf.argMax(byt, 1);
          const eq = tf.equal(pred, goal);
          const right = tf.sum(eq);
          const nright = right.dataSync()[0];

          num_right += nright;
          num_wrong += BATCH_SIZE - nright;
          num += BATCH_SIZE;
        });
    }
    const accuracy = num_right / (num_right + num_wrong);
    accuracys.push({x: epoch, y: accuracy});
    console.log("VAL", num_right, num_wrong, accuracy, num);
    // end validation

    {
      const series = ['Accuracy'];      
      const data = { values: [accuracys], series};
      const surface = tfvis.visor().surface({ tab: 'Validation', name: 'Accuracy',
                                            styles: { height: 200, maxHeight: 200}});
      tfvis.render.linechart(data, surface);
    }
    {
      const series = ['Loss'];
      const data = { values: [losses], series };
      const surface = tfvis.visor().surface({ tab: 'Training', name: 'Epoch/Loss',
                                            styles: { height: 200, maxHeight: 200}});
      tfvis.render.linechart(data, surface);
    }
    {
      const series = ['Times'];
      const data = { values: [times], series };
      const surface = tfvis.visor().surface({ tab: 'Training', name: 'Epoch/Time',
                                            styles: { height: 200, maxHeight: 200}});
      tfvis.render.linechart(data, surface);
    }    

    console.log("EPOCH FINISHED", epoch, "sum_loss", sum_loss, tf.memory(), (t2-t1));    
  }
  console.log("TRAINING FINISHED");
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

function toggle(but, id) {
  var el = document.getElementById(id);

  if (el.style.display === 'none') {
    el.style.display = 'block';
    but.innerHTML = SHOW;
  } else {
    el.style.display = 'none';
    but.innerHTML = HIDE;
  }
  console.log(but);
}

function show(but, id) {
  document.getElementById(id).style.display = 'block';
  console.log(but);
}

function hide(but, id) {
  document.getElementById(id).style.display = 'none';
}

async function init() {
  truncatedMobileNet = await loadTruncatedMobileNet();
  tfvis.visor().close();
  const surface = { tab: 'Model Summary', name: 'Truncated MobileNet' };
  tfvis.show.modelSummary(surface, truncatedMobileNet);

  const s = { tab: 'Details', name: 'conv_pw_13_relu'};
  tfvis.show.layer(s, truncatedMobileNet.getLayer('conv_pw_13_relu'));
}

// Initialize the application.
init();


