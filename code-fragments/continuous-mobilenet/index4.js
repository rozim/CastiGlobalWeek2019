

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
  if (worse) {
    return;
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
  } else{
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
