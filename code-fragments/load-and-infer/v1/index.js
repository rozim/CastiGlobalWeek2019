const predictionsElement = document.getElementById('predictions');
const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;
const MOBILENET_MODEL_PATH = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';
    
const IMAGE_SIZE = 224;
const TOPK_PREDICTIONS = 10;

import {IMAGENET_CLASSES} from './imagenet_classes';
var mobilenet;

const mobilenetDemo = async () => {
    console.log('XXX mobilenetDemo');
    status('loading');
    mobilenet = await tf.loadModel(MOBILENET_MODEL_PATH);
    console.log("xmobile", mobilenet);
    mobilenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
    status('');
    const catElement = document.getElementById('cat');
    if (catElement.complete && catElement.naturalHeight !== 0) {
	predict(catElement);
	catElement.style.display = '';
    } else {
	catElement.onload = () => {
	    predict(catElement);
	    catElement.style.display = '';
	}
    }
}

async function predict(imgElement) {
    console.log('XXX predict');
    status('Predicting...');

    const startTime = performance.now();
    const logits = tf.tidy(() => {
	    // tf.fromPixels() returns a Tensor from an image element.
	    const img = tf.fromPixels(imgElement).toFloat();

	    const offset = tf.scalar(127.5);
	    // Normalize the image from [0, 255] to [-1, 1].
	    const normalized = img.sub(offset).div(offset);

	    // Reshape to a single-element batch so we can pass it to predict.
	    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

	    // Make a prediction through mobilenet.
	    return mobilenet.predict(batched);
	});

    // Convert logits to probabilities and class names.
    const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
    const totalTime = performance.now() - startTime;
    status(`Done in ${Math.floor(totalTime)}ms`);

    // Show the classes in the DOM.
    showResults(imgElement, classes);
}

/**
 * Computes the probabilities of the topK classes given logits by computing
 * softmax to get probabilities and then sorting the probabilities.
 * @param logits Tensor representing the logits from MobileNet.
 * @param topK The number of top predictions to show.
 */
async function getTopKClasses(logits, topK) {
    console.log('XXX getTopKClasses');
    const values = await logits.data();

    const valuesAndIndices = [];
    for (let i = 0; i < values.length; i++) {
	valuesAndIndices.push({value: values[i], index: i});
    }
    valuesAndIndices.sort((a, b) => {
	    return b.value - a.value;
	});
    const topkValues = new Float32Array(topK);
    const topkIndices = new Int32Array(topK);
    for (let i = 0; i < topK; i++) {
	topkValues[i] = valuesAndIndices[i].value;
	topkIndices[i] = valuesAndIndices[i].index;
    }

    const topClassesAndProbs = [];
    for (let i = 0; i < topkIndices.length; i++) {
	topClassesAndProbs.push({
		className: IMAGENET_CLASSES[topkIndices[i]],
		    probability: topkValues[i]
		    })
	    }
    return topClassesAndProbs;
}

function showResults(imgElement, classes) {
    console.log('XXX showResults');
    const predictionContainer = document.createElement('div');
    predictionContainer.className = 'pred-container';

    const imgContainer = document.createElement('div');
    imgContainer.appendChild(imgElement);
    predictionContainer.appendChild(imgContainer);

    const probsContainer = document.createElement('div');
    for (let i = 0; i < classes.length; i++) {
	const row = document.createElement('div');
	row.className = 'row';

	const classElement = document.createElement('div');
	classElement.className = 'cell';
	classElement.innerText = classes[i].className;
	row.appendChild(classElement);

	const probsElement = document.createElement('div');
	probsElement.className = 'cell';
	probsElement.innerText = classes[i].probability.toFixed(3);
	row.appendChild(probsElement);

	probsContainer.appendChild(row);
    }
    predictionContainer.appendChild(probsContainer);

    predictionsElement.insertBefore(
				    predictionContainer, predictionsElement.firstChild);
}

const previewFile = async () => {
    console.log('XXX previewFile');
    // var preview = document.querySelector('img'); //selects the query named img
    var preview = document.querySelector('.myclass');
    var file    = document.querySelector('input[type=file]').files[0]; //sames as here
    var reader  = new FileReader();

    reader.onloadend = function () {
	preview.src = reader.result;
    }

    if (file) {
	reader.readAsDataURL(file); //reads the data as a URL
    } else {
	preview.src = "";
    }
}

    
console.log('XXX before');
mobilenetDemo();
console.log('XXX after');