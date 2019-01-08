
const video = document.querySelector("#videoElement");
/*
video.addEventListener('play', () => {
    console.log("play...");
    window.setInterval(grabFrame, 1000);
  }
  );
*/

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({video: true})
  .then(function(stream) {
    video.srcObject = stream;
  })
  .catch(function(err0r) {
    console.log("Something went wrong!");
  });
}

function grabFrame() {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  //var myImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
}

