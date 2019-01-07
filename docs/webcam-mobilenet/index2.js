

const video = document.querySelector("#videoElement");
video.addEventListener('play', () => {
    window.setInterval(step, 1000);
  }
);

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let  go_on = true;

const pmap = new Map();

if (navigator.mediaDevices.getUserMedia) {       
    navigator.mediaDevices.getUserMedia({video: true})
  .then(function(stream) {
    video.srcObject = stream;
  })
  .catch(function(err0r) {
    console.log("Something went wrong!");
  });
} 
let hack;
let last = new Date().getTime();
let waiting = false;

let tick = 0;
let mymodel = null;

function step() {
  if (! go_on) {
    return;
  }
  //requestAnimationFrame(step);

  tick += 1;
  var now = new Date().getTime();
  if ((now - last) < 5000) {
    document.getElementById('status').innerHTML = 5 - Math.round((now - last)/1000);
    return;
  }
  if (waiting) {
    document.getElementById('status').innerHTML = 'busy';
    return;
  }
  document.getElementById('status').innerHTML = '';
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  var myImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);    
  console.log(tick, canvas.width, canvas.height, now-last, myImageData);

  const img = tf.fromPixels(canvas).toFloat();

  waiting = true;

  for (let i = 0; i < 3; i++) {
    document.getElementById("c" + i).innerHTML = '?';
    document.getElementById("p" + i).innerHTML = '??';
    document.getElementById("x" + i).value = 0;
  }

  if (mymodel == null) {
    mobilenet.load().then(model => {
        mymodel = model;
        do_classify(img);
      });
  } else {
    do_classify(img);
  }
  /*
  mobilenet.load().then(model => {
      let t2 = new Date().getTime();        
      model.classify(img).then(predictions => {
          let t3 = new Date().getTime();
          console.log('TIME', t3-t2, t2-t1);
          showPredictions(predictions);
          last = new Date().getTime();
          waiting = false;
        })});
  */
}

function do_classify(img) {
  let t2 = new Date().getTime();        
  mymodel.classify(img).then(predictions => {
      let t3 = new Date().getTime();
      console.log('TIME', t3-t2);
      showPredictions(predictions);
      last = new Date().getTime();
      waiting = false;
    });
}



function speak() {
  speechSynthesis.speak(new SpeechSynthesisUtterance("hi"));
  document.getElementById('speak-button').style.display = 'none';
}

function showPredictions(predictions) {
  hack = predictions;
  console.log('Predictions: ');
  console.log(predictions);
  const bestClassName = predictions[0].className;
  const bestProbability = Math.round(100.0 * predictions[0].probability);

  let better = false;
  let worse = false;
  if (pmap.has(bestClassName)) {
    var old = pmap.get(bestClassName);
    if (old < bestProbability) {
      console.log("BETTER", bestClassName, old, bestProbability);      
      better = true;
    } else {
      worse = true;
    }
  }

  for (var i = 0; i < 3; i++) {
    const className = predictions[i].className;
    const probability = Math.round(predictions[i].probability * 100.0);
    document.getElementById("c" + i).innerHTML = className;
    document.getElementById("p" + i).innerHTML = probability + "%";
    document.getElementById("x" + i).value = probability;
  }
  if (worse || bestProbability < 10) {
    return;  // Need more confidence or improvement.
  }
  // OK: better or new.

  pmap.set(bestClassName, bestProbability);
  if (better) {
    window.speechSynthesis.speak(new SpeechSynthesisUtterance("better " + bestClassName));
  } else {
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(bestClassName));
  }

  // Make class name shorter for display.
  comma = bestClassName.indexOf(',');
  if (comma > 0) {
    cn = bestClassName.substring(0, comma);
  } else {
    cn = bestClassName;
  }

  const xid = "id-" + cn;
  let update = true;
  let newdiv = document.getElementById(xid);

  if (newdiv == null) {
    update = false;
    console.log('new', xid);
    newdiv = document.createElement("div");
    newdiv.className = "x-div";
    newdiv.id = xid;
    newdiv.bestClassName = bestClassName;    
  } else {
    console.log('update', xid);    
    newdiv.innerHTML = "";
  }

  var newimg = document.createElement("img");
  newimg.addEventListener('click', () => {
      console.log('click: ', newimg, newdiv);
      pmap.delete(newdiv.bestClassName);      
      newdiv.remove();
    });
  newimg.className = "x-img";
  newimg.src = canvas.toDataURL();
  newimg.width = newimg.height = 128;
  
  var newimgdiv = document.createElement("div");
  newimgdiv.className = "x-img-wrap";
  newimgdiv.appendChild(newimg);
  
  newdiv.appendChild(newimgdiv);
  newdiv.appendChild(document.createElement("br"));
  const text = document.createElement("span");
  text.className = "x-text";  
  text.appendChild(document.createTextNode(cn + "(" + bestProbability + "%)"));
  newdiv.appendChild(text);
  if (update) {
  } else {
    document.getElementById("images").appendChild(newdiv);
  }
}

// Great artists copy :)
// https://github.com/tensorflow/tfjs-examples/blob/master/simple-object-detection/train.js

// Name prefixes of layers that will be unfrozen during fine-tuning.
const topLayerGroupNames = ['conv_pw_9', 'conv_pw_10', 'conv_pw_11'];

// Name of the layer that will become the top layer of the truncated base.
const topLayerName =
    `${topLayerGroupNames[topLayerGroupNames.length - 1]}_relu`;

function buildNewHead(inputShape) {
  const newHead = tf.sequential();
  newHead.add(tf.layers.flatten({inputShape}));
  newHead.add(tf.layers.dense({units: 200, activation: 'relu'}));
  // Five output units:
  //   - The first is a shape indictor: predicts whether the target
  //     shape is a triangle or a rectangle.
  //   - The remaining four units are for bounding-box prediction:
  //     [left, right, top, bottom] in the unit of pixels.
  newHead.add(tf.layers.dense({units: 5}));  // <-- TBD 2
  return newHead;
}

function customLossFunction(yTrue, yPred) {
  return tf.tidy(() => {
      // Scale the the first column (0-1 shape indicator) of `yTrue` in order
      // to ensure balanced contributions to the final loss value
      // from shape and bounding-box predictions.
      return tf.metrics.meanSquaredError(yTrue.mul(LABEL_MULTIPLIER), yPred);
    });
}

async function buildObjectDetectionModel() {
  const {truncatedBase, fineTuningLayers} = await loadTruncatedBase();

  // Build the new head model.
  const newHead = buildNewHead(truncatedBase.outputs[0].shape.slice(1));
  const newOutput = newHead.apply(truncatedBase.outputs[0]);
  const model = tf.model({inputs: truncatedBase.inputs, outputs: newOutput});

  return {model, fineTuningLayers};
}

async function loadTruncatedBase() {
  const mobilenet = await tf.loadModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  console.log('mobilenet', mobilenet);
  mobilenet.summary();

  // Return a model that outputs an internal activation.
  const fineTuningLayers = [];

  // TBD: this is conv_pw_11_relu
  // TBD: may want conv_preds instead, and no fine tuning layers
  const layer = mobilenet.getLayer(topLayerName);
    const truncatedBase =
        tf.model({inputs: mobilenet.inputs, outputs: layer.output});
    // Freeze the model's layers.
    for (const layer of truncatedBase.layers) {
      layer.trainable = false;
      for (const groupName of topLayerGroupNames) {
        if (layer.name.indexOf(groupName) === 0) {
          fineTuningLayers.push(layer);
          break;
        }
      }
    }

    tf.util.assert(
        fineTuningLayers.length > 1,
        `Did not find any layers that match the prefixes ${topLayerGroupNames}`);
    return {truncatedBase, fineTuningLayers};
}

async function doit() {
  const {model, fineTuningLayers} = await buildObjectDetectionModel();
  console.log('doit', model, fineTuningLayers);
  model.compile({loss: customLossFunction, optimizer: tf.train.rmsprop(5e-3)});
  model.summary();

  // later need
  //  for (const layer of fineTuningLayers) {
  //layer.trainable = true;
  //}
  // also need to enable softmax
}

