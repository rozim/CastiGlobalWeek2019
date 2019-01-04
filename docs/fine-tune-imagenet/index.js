
document.getElementById('select1').onchange = function(evt) {
  doit(this.files, document.querySelector('#preview1'));
}

document.getElementById('select2').onchange = function(evt) {
  doit(this.files, document.querySelector('#preview2'));
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


