/* API Reference : https://js.tensorflow.org/api/1.0.0/ */


/* Constants */
//Image size supported by MobileNet
const MOBILE_NET_HEIGHT = 224;
const MOBILE_NET_WIDTH = 224;
const MOBILE_NET_PIXEL_PARTS = 3;

//MobileNet location in JSON format
const MOBILE_NET_REMOTE_ADDRESS = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json';
const MOBILE_NET_LOCAL_ADDRESS = 'indexeddb://mobilenet_v1_1.0_224.json';

//Classes
const NB_CLASSES = 3;
const LEFT = 'left';
const STRAIGHT = 'straight';
const RIGHT = 'right';


/* Classes converter */
//Name to id
const CLASS_NAME_TO_ID = {};
CLASS_NAME_TO_ID[LEFT] = 0;
CLASS_NAME_TO_ID[RIGHT] = 1;
CLASS_NAME_TO_ID[STRAIGHT] = 2;

//Id to name
const CLASS_ID_TO_NAME = {};
CLASS_ID_TO_NAME[0] = LEFT;
CLASS_ID_TO_NAME[1] = RIGHT;
CLASS_ID_TO_NAME[2] = STRAIGHT;


/* On ready */
$(document).ready(async () => {

	//Tell to Tensorflow to use WebGL and run in safe mode.
	tf.setBackend('webgl', true);

	//Get the pretrained neural network.
	const fullPretrained = await getPretrainedNeuralNetwork();

	//Truncate the pretrained neural network to remove the end, not adapted to our problem.
	const pretrained = truncateNeuralNetwork(fullPretrained, 'conv_pw_13_relu');

	//Store examples processed with the beginning of our neural network and the corresponding class given by user.
	let examplesData;
	let examplesClasses;
	//The example handler is called when an example is given by the user
	UI.exampleHandler = (direction) => {

		//Encode the class of the example in a neural network understandable format.
		const tensorClass = encodeClass(direction);

		//Get the webcam current image.
		const image = Webcam.webcamImage;

		//Encode the image of the example in a neural network understandable format.
		const tensorImage = encodeImage(image);

		//Get pretrained neural network output.
		const pretrainedOutput = pretrained.predict(tensorImage);

		//Save images and classes.
		if (examplesData !== undefined) {//If examplesData is set (we are not processing the first example).

			//Concat data.
			examplesData = examplesData.concat(pretrainedOutput, 0);

			//Concat classes.
			examplesClasses = examplesClasses.concat(tensorClass, 0);
		} else {
			//Initialize with first exemple.
			examplesData = pretrainedOutput;
			examplesClasses = tensorClass;
		}
	};

	//Learn with examples.
	let myKinect;//The end of our neural network.
	//The learn handler is called when the user want to start the learning
	UI.learnHandler = async (trainCallbacks) => {

		if (myKinect) {
			//Remove old myKinect neural network from GPU memory.
			myKinect.dispose();
		}

		//Get our neural network.
		myKinect = buildMyKinect(pretrained.output.shape, Hyperparameters.hiddenUnits, NB_CLASSES);

		//Define the batch size
		const batchSize = Math.floor(examplesData.shape[0] * Hyperparameters.batchSize);

		//Train the end of our neural network
		await myKinect.fit(examplesData, examplesClasses, {
			batchSize: batchSize,
			epochs: Hyperparameters.epochs,
			callbacks: trainCallbacks
		});
	};

	//Predict and update car direction wanted by user.
	//Called when the game want to update the car direction
	UI.updateCarDirection = async () => {

		//Get the webcam current image.
		const image = Webcam.webcamImage;

		//Encode the image in a neural network understandable format.
		const tensorImage = encodeImage(image);

		//Get pretrained output.
		const pretrainedOutput = pretrained.predict(tensorImage);

		//Predict probability for each class (direction).
		const predictions = myKinect.predict(pretrainedOutput);

		//Get the index of the higher probability.
		const predictedClassTensor = predictions.as1D().argMax();

		//Get the JavaScript value.
		const predictedClassId = (await predictedClassTensor.data())[0];

		//Get predicted class name.
		const predictedClassName = CLASS_ID_TO_NAME[predictedClassId];

		//Update car direction.
		if (predictedClassName === LEFT) {
			Car.goLeft();
		} else if (predictedClassName === RIGHT) {
			Car.goRight();
		} else if (predictedClassName === STRAIGHT) {
			Car.goStraight();
		}

		//Clear GPU memory.
		tensorImage.dispose();
		pretrainedOutput.dispose();
		predictions.dispose();
		predictedClassTensor.dispose();
	};
});


/* Functions */
/**
 * Loads and returns the pretrained model.
 */
async function getPretrainedNeuralNetwork() {
	let mobilenet;
	try {
		console.log('Loading MobileNet from local storage...');

		//To avoid downloading MobileNet again, we check if we have already stored it locally
		mobilenet = await tf.loadLayersModel(MOBILE_NET_LOCAL_ADDRESS);

		console.log('MobileNet loaded from local storage');
	} catch (error) {
		console.log('MobileNet not found in local storage, downloading it...');

		//We download MobileNet and load it into memory
		mobilenet = await tf.loadLayersModel(MOBILE_NET_REMOTE_ADDRESS);

		console.log('MobileNet loaded from remote, saving it');

		//To avoid downloading MobileNet again when reloading the page, we save it locally
		await mobilenet.save(MOBILE_NET_LOCAL_ADDRESS);
	}
	return mobilenet;
}

/**
 * Returns a truncated representation of a neural network.
 *
 * @param neuralNetwork The neural network to truncate.
 * @param endLayerName The name of the last layer to keep.
 */
function truncateNeuralNetwork(neuralNetwork, endLayerName) {

	//Get the end layer by name.
	const endLayer = neuralNetwork.getLayer(endLayerName);

	//Build the truncated neural network. The new model is defined by original inputs and the output of the end layer.
	return tf.model({
		inputs: neuralNetwork.inputs,
		outputs: endLayer.output
	});
}

/**
 * Converts a class name into something understandable for a neural network.
 *
 * @param className The class name to convert.
 */
function encodeClass(className) {
	//We will create intermediate tensors, we have to be sure that they will all be disposed from GPU memory.
	return tf.tidy(() => {

		//Encode the class name into a number.
		const classId = CLASS_NAME_TO_ID[className];

		//Convert the class number into an int32 tensor.
		const classIdTensor = tf.scalar(classId, 'int32');

		//Encode the class in a one-hot representation.
		const oneHotClass = tf.oneHot(classIdTensor, NB_CLASSES);

		//In TensorFlow, neural networks handle batched data, so we have to transform the class into a batch of one class.
		const batchedClass = oneHotClass.expandDims(0);

		return batchedClass;
	});
}

/**
 * Converts an image into something understandable for a neural network.
 *
 * @param image The image to convert.
 */
function encodeImage(image) {
	//We will create a lot of intermediate tensors, we have to be sure that they will all be disposed from GPU memory.
	return tf.tidy(() => {

		//Read and convert image into a tensor. TensorFlow.js can directly import pixels in an HTML element as a tensor.
		const tfImage = tf.browser.fromPixels(image);

		//Crop image to fit the pretrained model input size.
		const croppedImage = TFUtil.cropImage(tfImage);

		//In TensorFlow, neural networks handle batched data, so we have to transform the image into a batch of one image.
		const batchedImage = croppedImage.expandDims(0);

		//tf.fromPixels converts an image into a int32 tensor, but MobileNet handles float32, so we need to convert our image into float.
		const floatImage = batchedImage.toFloat();

		//MobileNet handles normalized input (between 1.0 and -1.0), but images are encoded with values between 0 and 255.
		//To normalize our image data, we need to divide each value by 127 then subtract 1.
		const normalizedImage = floatImage.div(tf.scalar(127.0)).sub(tf.scalar(1.0));

		return normalizedImage;
	});
}

/**
 * Builds a new neural network with 1 hidden layer that will be the end of our model.
 *
 * @param inputShape The input size.
 * @param hiddenUnits The number of units in the hidden layer.
 * @param outputUnits The number of units in the output layer.
 */
function buildMyKinect(inputShape, hiddenUnits, outputUnits) {
	//Neural network layers definition.
	const layers = [
		tf.layers.flatten({batchInputShape: inputShape}),//Layer to flatten pretrained neural network output (format in 2 dimensions).
		tf.layers.dense({//Hidden layer, fully connected, we will use 'relu' activation, 'varianceScaling' kernelInitializer and bias
			units: hiddenUnits,
			activation: 'relu',
			kernelInitializer: 'varianceScaling',
			useBias: true
		}),
		tf.layers.dense({//Output layer, fully connected, we will use 'softmax' activation, 'varianceScaling' kernelInitializer and not bias
			units: outputUnits,
			activation: 'softmax',
			kernelInitializer: 'varianceScaling',
			useBias: false
		})
	];

	//Create sequential neural network (stack of layers).
	const model = tf.sequential({
		layers: layers
	});

	//Create optimizer.
	const optimizer = tf.train.adam(Hyperparameters.learningRate);

	//Compile model.
	model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

	return model;
}
