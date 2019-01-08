
let lfd = null;
let ffd = null;
let fresults = null;

const NUM = 100;
const maxDescriptorDistance = 1.0;
const CANVAS_SIZE = 256;

// This will have all our canned images.
// [label, [descriptors]]
const labeledFaceDescriptors = new Array();

document.getElementById("upload").onchange = function(evt) {
  for (let i = 0; i < this.files.length; i++) {
    readAndPreview(this.files[i]);
  }
  function readAndPreview(file) {
    ImageTools.resize(file, {
       width: CANVAS_SIZE,
       height: CANVAS_SIZE
     }, function(blob, didItResize) {
        const image = element("upload-preview");
        //image.id = ("img_" + cn + "_" + seq);
        //image.className = "train-image";
        image.src = URL.createObjectURL(blob);
        image.title = file.name;
        //image.width = CANVAS_SIZE;
        image.height = CANVAS_SIZE;

        const onload = (my_image) => (event) => {

          munch_image(my_image, "upload-best");
          /*
          faceapi.detectAllFaces(my_image).withFaceLandmarks().withFaceDescriptors().then((fullFaceDescriptions) => {
              console.log("onload", fullFaceDescriptions);
              const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, maxDescriptorDistance);
              const results = fullFaceDescriptions.map(fd => faceMatcher.findBestMatch(fd.descriptor));
              const url = "celebrities/" + results[0].label + ".jpg";

              let best = element("upload-best");
              best.src = url;
              best.width = best.height = 256;
              best.title = results[0].label;
            }
            );
          */
        }
        image.onload = onload(image);
      }
      );
  };
};


function upload(url) {
  console.log("upload", url);
  element("upload-best").src = "";
  let preview = element("upload-preview");
  preview.src = url;
  preview.height = CANVAS_SIZE;
  preview.onload = function() {
    munch_image(preview, "upload-best");
  };
}


async function init() {
  await faceapi.loadSsdMobilenetv1Model('models/');
  await faceapi.loadFaceLandmarkModel('models/');
  await faceapi.loadFaceRecognitionModel('models/');

  const container = document.getElementById("container");
  const pro = document.getElementById("pro");
  pro.value = 0;

  shuffleArray(CELEBS);
  const hack = CELEBS.slice(0, NUM);
  pro.max = hack.length;

  const t1 = new Date().getTime();
  for (let i = 0; i < hack.length; i++) {
    pro.value = i+1;
    const label = hack[i];
    const t1 = new Date().getTime();
    const full = "celebrities/" + label;
    const imgUrl = `${full}.jpg`;
    const img = await faceapi.fetchImage(imgUrl);

    const fullFaceDescription = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();

    if (!fullFaceDescription) {
      console.log(`no faces detected for ${label} at ${imgUrl}`);
      continue;
    }

    const link = document.createElement("a");
    img.height = 96;
    img.title = label;
    img.border = 0;
    link.appendChild(img);
    link.href = imgUrl;
    link.target = "_blank";
    link.title = label;
    container.appendChild(link);

    const faceDescriptors = [fullFaceDescription.descriptor];
    if (i % 10 == 0) {
      console.log(label, "done", ((new Date().getTime() - t1) / 1000.0).toFixed(2));
    }
    labeledFaceDescriptors.push(new faceapi.LabeledFaceDescriptors(label, faceDescriptors));
  }

  console.log("OK after", ((new Date().getTime() - t1) / 1000.0).toFixed(2));

  lfd = labeledFaceDescriptors;

  element("loading").innerHTML = "Loaded " + labeledFaceDescriptors.length;

  /*
  cur_ffd = cur_ffd.filter(function(value, index, arr){
      return value.label != results[0].label;
    });
  */
}

function shuffleArray(a1) {
  for (let i = a1.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a1[i], a1[j]] = [a1[j], a1[i]];
  }
}

function element(id) {
  return document.getElementById(id);
}

function munch_image(my_image, best_id) {
  faceapi.detectAllFaces(my_image).withFaceLandmarks().withFaceDescriptors().then((fullFaceDescriptions) => {
      console.log("onload", fullFaceDescriptions);
      const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, maxDescriptorDistance);
      const results = fullFaceDescriptions.map(fd => faceMatcher.findBestMatch(fd.descriptor));
      const url = "celebrities/" + results[0].label + ".jpg";

      let best = element(best_id);
      best.src = url;
      best.height = CANVAS_SIZE;
      best.title = results[0].label;
    }
    )
      };


let remain = 99;
function start_video_countdown() {

  const id = element("video-countdown");

  remain = 4;
  decr();

  function decr() {
    remain -= 1;

    if (remain > 0) {
      id.innerHTML = remain;
      window.setTimeout(decr, 1000);
    } else {
      id.innerHTML = "";
      grabFrame();
      munch_image(video, "webcam-best");
      id.InnerHTML = "";
    }
  }
}



init();