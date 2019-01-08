
let lfd = null;
let ffd = null;
let fresults = null;

const NUM = 100;
const maxDescriptorDistance = 1.0;

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

  const labeledFaceDescriptors = new Array();
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
    link.title = label;
    container.appendChild(link);

    const faceDescriptors = [fullFaceDescription.descriptor];
    console.log(label, "done", ((new Date().getTime() - t1) / 1000.0).toFixed(2));
    labeledFaceDescriptors.push(new faceapi.LabeledFaceDescriptors(label, faceDescriptors));
  }

  console.log("OK after", ((new Date().getTime() - t1) / 1000.0).toFixed(2));

  lfd = labeledFaceDescriptors;

  const input = document.getElementById("nixon");
  let fullFaceDescriptions = await faceapi.detectAllFaces(input).withFaceLandmarks().withFaceDescriptors();
  ffd = fullFaceDescriptions;

  let cur_ffd = labeledFaceDescriptors.slice(0); // clone

  for (let num = 0; num < 3; num++) {
    //console.log("cur", cur_ffd);
    //const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, maxDescriptorDistance);
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
    best.appendChild(img);

    cur_ffd = cur_ffd.filter(function(value, index, arr){
        return value.label != results[0].label;
      });
  }
}

function shuffleArray(a1) {
  for (let i = a1.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a1[i], a1[j]] = [a1[j], a1[i]];
  }
}


init();