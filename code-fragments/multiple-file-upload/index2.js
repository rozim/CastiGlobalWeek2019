
document.getElementById('select').onchange = function(evt) {
    var preview = document.querySelector('#preview');
  
    if (this.files) {
	[].forEach.call(this.files, readAndPreview);
    }

    function readAndPreview(file) {
	console.log("LOAD ", file);
	ImageTools.resize(file, {
		width: 64, 
		height: 64 
	    }, function(blob, didItResize) {
		var image = new Image();
		// image.src = 'data:image/bmp;base64,' + btoa(blob); 
		image.src = URL.createObjectURL(blob);
		image.title = file.name;
		preview.appendChild(image);

		console.log("file", file);		
		console.log("image", image);
		console.log("blob", blob);		
		console.log("blob", URL.createObjectURL(blob));
		console.log("btoa", btoa(blob))
		console.log("atob", btoa(blob))				    		
	    }
	    )}
};


