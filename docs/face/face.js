

//let fd = null;
let lfd = null;
let ffd = null;
let fresults = null;

async function init() {
  await faceapi.loadSsdMobilenetv1Model('models/');
  await faceapi.loadFaceLandmarkModel('models/');
  await faceapi.loadFaceRecognitionModel('models/');
  
  const container = document.getElementById("container");

  const labels = [ "bush", "carter", "melania", "obama", "trump"];
  //const labels = [ "bush", "nixon"];
  const labeledFaceDescriptors = await Promise.all(
      labels.map(async label => {
          const t1 = new Date().getTime();
          console.log(label);
          // fetch image data from urls and convert blob to HTMLImage element
          const full = "images/" + label;
          const imgUrl = `${full}.jpg`;
          const img = await faceapi.fetchImage(imgUrl);

          // detect the face with the highest score in the image and compute it's landmarks and face descriptor
          const fullFaceDescription = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();

          if (!fullFaceDescription) {
            throw new Error(`no faces detected for ${label}`)
          }

          img.width = img.height = 64;
          container.appendChild(img);          

          const faceDescriptors = [fullFaceDescription.descriptor];
          console.log(label, new Date().getTime() - t1);
          return new faceapi.LabeledFaceDescriptors(label, faceDescriptors)
        }));

  lfd = labeledFaceDescriptors;

  const input = document.getElementById("nixon");
  let fullFaceDescriptions = await faceapi.detectAllFaces(input).withFaceLandmarks().withFaceDescriptors();
  ffd = fullFaceDescriptions;
      
  //const maxDescriptorDistance = 0.6;
  const maxDescriptorDistance = 1.0;
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, maxDescriptorDistance);

  const results = await fullFaceDescriptions.map(fd => faceMatcher.findBestMatch(fd.descriptor));

  console.log("results", results);
  console.log("results", results[0].toString());
  fresults = results;                                                    
}


init();