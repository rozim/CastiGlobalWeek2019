
let mymodel = null;
const CANVAS_SIZE = 224;  // Matches the input size of MobileNet.

let fullMobileNet = null;
const normalizationOffset = tf.scalar(127.5);

const rows = new Map(); // id -> element
const images = new Map();  // id -> []
const predictions = new Map();  // id -> []
const freq = new Map();

document.getElementById("select1").onchange = function(evt) {
  addUploadedImages(this.files, document.querySelector("#preview1"), 0);
};

function addUploadedImages(files, preview, classNumber) {
  const p = document.getElementById("p_label" + (classNumber + 1));
  p.max = files.length;
  p.value = 0;

  for (let i = 0; i < files.length; i++) {
    readAndPreview(p, files[i], classNumber, i);
  }

  function readAndPreview(pro, file, cn, seq) {
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
          tf.tidy( () => { 
              const resized = imageToTensor(my_image);
              const batched = resized.reshape([1, CANVAS_SIZE, CANVAS_SIZE, 3]);
              const softmaxes = fullMobileNet.predict(batched).dataSync();
              const index = tf.argMax(softmaxes).dataSync()[0];
              const prediction = softmaxes[index];
              const pretty = simplify(IMAGENET[index]);
              const row_id = "row_" + pretty;
              let row;
              if (!images.has(row_id)) {
                console.log("NEW ROW", row_id);
                images.set(row_id, []);
                predictions.set(row_id, []);                
                freq.set(row_id, 0);
              }
              images.get(row_id).push(my_image);
              predictions.get(row_id).push(prediction);
              freq.set(row_id, freq.get(row_id) + 1);
              pro.value += 1;
              
              if (pro.value == pro.max) {
                console.log("DONE");
                const sorted = sort_map(freq);
                for (var i = 0; i < sorted.length; i++) {
                  const row_id = sorted[i][0];
                  const n = sorted[i][1]; // freq

                  if (!rows.has(row_id)) {
                    const t = document.createElement("div");
                    t.className = "row-label";
                    t.innerHTML = "<b>" + row_id.substring(4) + " (" + images.get(row_id).length + ") " + "</b>";
                    
                    row = document.createElement("div");
                    row.appendChild(t);
                    row.id = row_id;
                    row.className = "scrolling-wrapper-flexbox";
                    const el = element("container");
                    el.appendChild(row);
                    rows.set(row_id, row);
                  } else {
                    row = rows.get(row_id);
                  }
                  const imgs = images.get(row_id);
                  const preds = predictions.get(row_id);
                  const tuples = [];
                  for (var j = 0; j < imgs.length; j++) {
                    tuples.push([preds[j], imgs[j]]);
                  }
                  tuples.sort(function(a, b) {
                      a = a[0];
                      b = b[0];

                      return a > b ? -1 : (a < b ? 1 : 0);
                    });
                  for (var j = 0; j < imgs.length; j++) {
                    const pred = Math.round(tuples[j][0] * 100.0) + "%";
                    const img = tuples[j][1];

                    const figure = document.createElement("figure");
                    figure.style.minWidth = "224px";
                    const figcaption = document.createElement("figcaption");
                    figcaption.innerHTML = pred;                    
                    figure.appendChild(img);
                    figure.appendChild(figcaption);
                    row.appendChild(figure);
                  }
                }
              }
            });
        };
        image.onload = onload(image);        
      });
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




function getNumber(id) {
  return Number(document.getElementById(id).value);
}

function getInteger(id) {
  return Math.round(Number(document.getElementById(id).value));
}

function element(id) {
  return document.getElementById(id);
}

function sort_count(arr) {
  var counts = {};

  for (var i = 0; i < arr.length; i++) {
    var num = arr[i];
    counts[num] = counts[num] ? counts[num] + 1 : 1;
  }
  return counts;
}

function sort_count2(obj) {
  var tuples = [];

  for (var key in obj) tuples.push([key, obj[key]]);

  tuples.sort(function(a, b) {
      a = a[1];
      b = b[1];

      return a > b ? -1 : (a < b ? 1 : 0);
    });
  return tuples;
}

function sort_map(myMap) {
  let tuples = [];
  for (var [key, value] of myMap) {
    tuples.push([key, value]);
  }

  tuples.sort(function(a, b) {
      a = a[1];
      b = b[1];

      return a > b ? -1 : (a < b ? 1 : 0);
    });
  return tuples;    
}

function simplify(s) {
  var x = s.indexOf(',');
  if (x > 0) {
    return s.substring(0, x);
  } else {
    return s;
  }
}

async function init() {
  fullMobileNet = await tf.loadModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
}
// Initialize the application.
init();


