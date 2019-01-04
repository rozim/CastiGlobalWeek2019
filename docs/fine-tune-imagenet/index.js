
let mymodel = null;

console.log("Loading");
mobilenet.load().then(model => {
    mymodel = model;
    console.log("Loaded");
  });

document.getElementById("select1").onchange = function(evt) {
  doit(this.files, document.querySelector("#preview1"));
}

document.getElementById("select2").onchange = function(evt) {
  doit(this.files, document.querySelector("#preview2"));
}

function doit(files, preview) {
  if (files) {
    [].forEach.call(files, readAndPreview);
  }

  function readAndPreview(file) {
    // console.log("LOAD ", file);
    ImageTools.resize(file, {
       width: 64,
       height: 64
     }, function(blob, didItResize) {
        let image = new Image();
        image.src = URL.createObjectURL(blob);
        image.title = file.name;
        image.className = "train-image";

        let div = document.createElement("div");
        div.className = "card";
        div.appendChild(image);
        preview.appendChild(div);
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
    mymodel.classify(img2).then(predictions => {
        console.log("pred: ", predictions);
      });
    
  }
}