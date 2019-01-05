// import * as tf from '@tensorflow/tfjs';

// https://github.com/tensorflow/tfjs-examples/blob/master/simple-object-detection/train.js

let mymodel = null;
const CANVAS_SIZE = 224;  // Matches the input size of MobileNet.
const NUM_CLASSES = 2;
let truncatedMobileNet = null;
const normalizationOffset = tf.scalar(127.5);
const BATCH_SIZE = 8;
const PICK = 0.80;
const DENSE_SIZE = 5;
let global_model2;

const HIDE = "&#9654;";
const SHOW = "&#9660;";

let allow_training = true;

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

  /*
    console.log(files, preview, classNumber);    
  if (files) {
    [].forEach.call(files, readAndPreview);
  }
  */

  const p = document.getElementById("p_label" + (classNumber + 1));
  p.max = files.length;
  p.value = 0;
  for (let i = 0; i < files.length; i++) {
    readAndPreview(p, files[i], classNumber, i);
  }

  function readAndPreview(pro, file, cn, seq) {
    //console.log("LOAD ", file);
    ImageTools.resize(file, {
       width: CANVAS_SIZE,
       height: CANVAS_SIZE
     }, function(blob, didItResize) {
        let image = new Image();
        image.id = ("img_" + cn + "_" + seq);
        image.src = URL.createObjectURL(blob);
        image.title = file.name;
        image.className = "train-image";
        image.width = CANVAS_SIZE;
        image.height = CANVAS_SIZE;        
        let div = document.createElement("div");
        div.className = "card";
        div.appendChild(image);
        preview.appendChild(div);

        const onload = (my_image) => (event) => {

          if (seq % 10 == 0) {          
            let prom = tf.time(() =>
                               tf.tidy( () => { 
                                   const resized = imageToTensor(my_image);
                                   const batched = resized.reshape([1, CANVAS_SIZE, CANVAS_SIZE, 3]);
                                   my_image.activation = truncatedMobileNet.predict(batched).dataSync();
                                   pro.value += 1;
                                 }
                                 )
                               );
            prom.then( (foo) => {
                console.log(foo, tf.memory());
              });
          } else {
            tf.tidy( () => { 
                const resized = imageToTensor(my_image);
                const batched = resized.reshape([1, CANVAS_SIZE, CANVAS_SIZE, 3]);
                my_image.activation = truncatedMobileNet.predict(batched).dataSync();
                pro.value += 1;
              });
          }
        };
        image.onload = onload(image);        
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


async function stopit() {
  allow_training = false;
}

async function loadit2() {
  allow_training = true;

  const layers = [
       // Flattens the input to a vector so we can use it in a dense layer. While
       // technically a layer, this only performs a reshape (and has no training
       // parameters).
       tf.layers.flatten({
        inputShape: truncatedMobileNet.outputs[0].shape.slice(1)
         })];

  for (let i = 1; i <= 3; i++) {
    const n = getInteger("hidden" + i);
    console.log("DENSE", i, n);
    if (n <= 0) {
      continue;
    }
    const dense = tf.layers.dense(
        {
       units: n,
       activation: 'relu',
       kernelInitializer: 'varianceScaling',
       useBias: true,
       name: ('my_dense' + i)
        });
    layers.push(dense);
  }
  layers.push(
      tf.layers.dense(
          {
                units: NUM_CLASSES,
                kernelInitializer: 'varianceScaling',
                useBias: false,
                activation: 'softmax',
                name: 'my_softmax'
                }));
  
  let model2 = tf.sequential({
   layers: layers
    });
  const surface = { tab: 'Model Summary', name: 'MyModel' };
  tfvis.show.modelSummary(surface, model2);
  
  global_model2 = model2;

  let s = { tab: 'Details', name: 'my_softmax'};
  tfvis.show.layer(s, model2.getLayer('my_softmax'));
  //s = { tab: 'Details', name: 'my_dense'};
  //tfvis.show.layer(s, model2.getLayer('my_dense'));  

  const lr = getNumber("learning_rate");
  console.log("LR", lr);
  const optimizer = tf.train.adam(lr);
  model2.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

  let xs_all = new Array();
  let ys_all = new Array();
  console.log("MEM Loop begin", tf.memory());
  let mod = 100;

  let num_images = 0;
  for (var label = 0; label < NUM_CLASSES; label++) {
    num_images += document.getElementById("preview" + (label + 1)).childNodes.length;
  }

  let cur_image = 0;

  console.log("NEW/1", tf.memory());
  for (var label = 0; label < NUM_CLASSES; label++) {
    const y = tf.tidy(
        () => tf.oneHot(tf.tensor1d([label]).toInt(), NUM_CLASSES));
    const ydata = y.dataSync();
    tf.dispose(y); 
    for (var seq = 0; ; seq++) {
      var img = document.getElementById("img_" + label + "_" + seq);
      if (img == null) {
        break;
      }
      xs_all.push(img.activation);
      ys_all.push(ydata);
    }
  }
  console.log("NEW/2", tf.memory());  

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
  p_epoch.max = getInteger("epochs");
  const p_batch = document.getElementById("p_batch");
  p_batch.max = xs_train.length / BATCH_SIZE;
  const losses = new Array();
  const times = new Array();
  const accuracys = new Array();
  let batch_number = 0;


  const el_accuracy = element("best_accuracy");
  const el_loss = element("best_loss");
  el_accuracy.innerHTML = "";
  el_loss.innerHTML = "";
  let best_accuracy = 0.0;
  let best_loss = null;

  for (var epoch = 0; epoch < p_epoch.max; epoch++) {
    p_epoch.value = epoch + 1;
    shuffleArrays(xs_train, ys_train);
    let sum_loss = 0.0;
    const t1 = new Date().getTime();
    for (var start = 0; start < stop; start += BATCH_SIZE) {
      if (!allow_training) {
        console.log("TRAINING STOPPED");
        return;
      }
      p_batch.value = start / BATCH_SIZE;
      const bx = xs_train.slice(start, start + BATCH_SIZE);
      const by = ys_train.slice(start, start + BATCH_SIZE);
  
      const bxt = tf.tidy( () => tf.concat(bx).as4D(BATCH_SIZE, 7, 7, 256));
      const byt = tf.tidy( () => tf.concat(by).asType('float32').as2D(BATCH_SIZE, NUM_CLASSES));

      await model2.trainOnBatch(bxt, byt).then(loss =>
        {
          sum_loss += loss;
          tf.dispose([bxt, byt]);
          batch_number += 1;
        });
    } // end of one epoch of training loop

    const t2 = new Date().getTime();
    times.push({x: epoch, y: (t2 - t1) / 1000.0});
    losses.push({x: epoch, y: sum_loss});

    if (best_loss === null || sum_loss < best_loss) {
      best_loss = sum_loss;
      el_loss.innerHTML = sum_loss.toFixed(4);
    }

    // begin validation
    const vstop = xs_validation.length % BATCH_SIZE == 0 ? xs_validation.length : xs_validation.length - BATCH_SIZE;

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
    if (accuracy > best_accuracy) {
      best_accuracy = accuracy;
      el_accuracy.innerHTML = accuracy.toFixed(2);
    }
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

    console.log("EPOCH FINISHED", epoch, "sum_loss", sum_loss, tf.memory(), Math.ceil((t2-t1)/1000.0));
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

function getNumber(id) {
  return Number(document.getElementById(id).value);
}

function getInteger(id) {
  return Math.round(Number(document.getElementById(id).value));
}

function element(id) {
  return document.getElementById(id);
}

// Initialize the application.
init();


