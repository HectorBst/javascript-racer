
//Car control
const Car = {
	goLeft: () => {
		keyRight = false;
		keyLeft = true;
	},
	goRight: () => {
		keyLeft = false;
		keyRight = true;
	},
	goStraight: () => {
		keyLeft = false;
		keyRight = false;
	},
	goFaster: () => {
		keySlower = false;
		keyFaster = true;
	},
	goSlower: () => {
		keyFaster = false;
		keySlower = true;
	}
};

//Hyperparameters
const Hyperparameters = {
	get learningRate() {
		return Number($('#learning-rate').val());
	},
	get epochs() {
		return Number($('#epochs').val());
	},
	get batchSize() {
		return Number($('#batch-size').val());
	},
	get hiddenUnits() {
		return Number($('#hidden-units').val());
	}
};

//UI
const UI = {
	exampleHandler: () => {},
	learnHandler: async () => {},
	updateCarDirection: async () => {}
};

//Webcam
const Webcam = {
	get webcamImage() {
		return document.getElementById('webcam');
	}
};
$(document).ready(async () => {
	const videoElement = document.getElementById('webcam');

	if (navigator.getUserMedia) {
		const stream = await navigator.mediaDevices.getUserMedia({video: true});
		videoElement.height = MOBILE_NET_HEIGHT;
		videoElement.width = stream.getVideoTracks()[0].getSettings().aspectRatio * MOBILE_NET_WIDTH;
		videoElement.srcObject = stream;
	}
});

//Util
const TFUtil = {
	cropImage: (tfImage) => {
		const size = Math.min(tfImage.shape[0], tfImage.shape[1]);
		const centerHeight = tfImage.shape[0] / 2;
		const beginHeight = centerHeight - (size / 2);
		const centerWidth = tfImage.shape[1] / 2;
		const beginWidth = centerWidth - (size / 2);
		return tfImage.slice([beginHeight, beginWidth, 0], [size, size, 3]);
	}
};

//Listeners
$(document).ready(() => {
	$('#start-game-button').one('click', async function () {
		this.disabled = true;
		Game.start();
		Car.goFaster();
		while (true) {
			await UI.updateCarDirection();
			await tf.nextFrame();
			await tf.nextFrame();
			await tf.nextFrame();
			await tf.nextFrame();
		}
	});
	$('.example-button').click(function () {
		$('.example-button').each((i,e) => e.disabled = true);
		UI.exampleHandler($(this).data('label'));
		$(this).closest('div').find('.nb').each((i,e) => $(e).text(Number($(e).text()) + 1));
		$('.example-button').each(async (i,e) => e.disabled = false);
		if (!$('.example-button').closest('div').find('.nb').map((i,e) => Number($(e).text())).is((i,e) => e === 0)) {
			$('#learn-button').each((i,e) => e.disabled = false);
		}
	});
	$('#learn-button').click(async function () {
		$('.example-button').each((i,e) => e.disabled = true);
		this.disabled = true;

		const visor = tfvis.visor();
		visor.open();
		visor.unbindKeys();
		const surface = visor.surface({
			name: 'Training'
		});
		const fitCallbacks = tfvis.show.fitCallbacks(surface, ['loss'], {height: 300, zoomToFitAccuracy: true});

		await UI.learnHandler(fitCallbacks);
		this.disabled = false;
		$('.example-button').each(async (i,e) => e.disabled = false);
		$(this).text('ReLearn');
	});
});
