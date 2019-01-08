
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
        let image = new Image();
        //image.id = ("img_" + cn + "_" + seq);
        //image.className = "train-image";
        image.src = URL.createObjectURL(blob);
        image.title = file.name;
        //image.width = CANVAS_SIZE;
        image.height = CANVAS_SIZE;

        const preview = element("upload-preview");
        let container = element("upload-container");
        if (container) {
          container.parentNode.removeChild(container);
        }
        container = document.createElement("div");
        container.appendChild(image);
        preview.appendChild(container);

        const onload = (my_image) => (event) => {

          faceapi.detectAllFaces(my_image).withFaceLandmarks().withFaceDescriptors().then((fullFaceDescriptions) => {
              console.log("onload", fullFaceDescriptions);
              const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, maxDescriptorDistance);
              const results = fullFaceDescriptions.map(fd => faceMatcher.findBestMatch(fd.descriptor));
              const url = "celebrities/" + results[0].label + ".jpg";

              let container = element("upload-best");
              let best = document.getElementById("upload-match");
              if (best) {
                best.parentNode.removeChild(best);
              }
              best = document.createElement("img");
              best.src = url;
              best.width = best.height = 256;
              best.title = results[0].label;
              container.appendChild(best);
            }
            );
        }
        image.onload = onload(image);
      }
      );
  };
};

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
    img.width = img.height = 64;
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

  const input = document.getElementById("nixon");
  let fullFaceDescriptions = await faceapi.detectAllFaces(input).withFaceLandmarks().withFaceDescriptors();
  ffd = fullFaceDescriptions;

  let cur_ffd = labeledFaceDescriptors.slice(0); // clone

  const faceMatcher = new faceapi.FaceMatcher(cur_ffd, maxDescriptorDistance);
  const results = await fullFaceDescriptions.map(fd => faceMatcher.findBestMatch(fd.descriptor));

  console.log("results", results);
  console.log("results", results[0]);
  console.log("results", results[0].toString());
  fresults = results;

  const url = "celebrities/" + results[0].label + ".jpg";
  const best = document.getElementById("best");
  const img = document.createElement("img");
  img.src = url;
  img.title = results[0].label;
  img.height = CANVAS_SIZE;
  best.appendChild(img);

  element("loading").style.display = "none";

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


init();