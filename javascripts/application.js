var canvas = document.getElementById("canvas");
var context = canvas.getContext("2d");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

var config = {
  structure: "2 5 7 3 1",
  eta: 0.5,
  momentum: 0.5,
  min_mse: 0.01,
  max_epochs: 100000
};
var decimalPlaces = 100;

var network = {
  layers: [],
  layersIndex: 0,
  construct: function() {
    network.clearCanvas();
    var layerStructure = config.structure.split(" ");
    var layerWidth = canvas.width / layerStructure.length;
    for (var i = 0; i < layerStructure.length; i++) {
      var type = "HIDDEN";
      if (i == 0) {
        type = "INPUT";
      } else if (i == layerStructure.length - 1) {
        type = "OUTPUT";
      }
      network.layers.push(new Layer(type));
      var layerSize = parseInt(layerStructure[i]);
      layerSize = i == 0 ? layerSize + 1 : layerSize;
      var layerGridHeight = canvas.height / layerSize;
      for (var j = 0; j < layerSize; j++) {
        var x = layerWidth * i + layerWidth / 2;
        var y = layerGridHeight * j + layerGridHeight / 2;
        network.layers[i].perceptrons.push(new Perceptron(x, y));
      }
    }
  },
  connect: function(data) {
    var left = network.layers[data.leftLayer].perceptrons[data.leftPerceptron];
    var right = network.layers[data.rightLayer].perceptrons[data.rightPerceptron];
    var connection = new Connection(left, right, data.weight);
    left.rightConnections.push(connection);
    right.leftConnections.push(connection);
    network.draw();
  },
  inputReceived: function(data) {
    var perceptron = network.layers[data.layer].perceptrons[data.perceptron];
    perceptron.inputReceived(network.layers[data.layer].type == "INPUT" ? Math.round(data.value * Math.pow(10, decimalPlaces)) / Math.pow(10, decimalPlaces) : "");
    for (var i = 0; i < perceptron.leftConnections.length; i++) {
      perceptron.leftConnections[i].feedForward();
    }
    network.draw();
  },
  feedForward: function(data) {
    var perceptron = network.layers[data.layer].perceptrons[data.perceptron];
    perceptron.feedForward(Math.round(data.value * Math.pow(10, decimalPlaces)) / Math.pow(10, decimalPlaces));
    for (var i = 0; i < perceptron.rightConnections.length; i++) {
      perceptron.rightConnections[i].inputReceived();
    }
    network.draw();
  },
  backPropagate: function(data) {
    var perceptron = network.layers[data.layer].perceptrons[data.perceptron];
    perceptron.backPropagate();
    for (var i = 0; i < perceptron.rightConnections.length; i++) {
      perceptron.rightConnections[i].backPropagate();
    }
    network.draw();
  },
  updateDelta: function(data) {
    var perceptron = network.layers[data.layer].perceptrons[data.perceptron];
    perceptron.updateDelta(Math.round(data.value * Math.pow(10, decimalPlaces)) / Math.pow(10, decimalPlaces));
    network.draw();
  },
  draw: function() {
    network.clearCanvas();
    for (var i = 0; i < network.layers.length; i++) {
      network.layers[i].draw();
    }
  },
  clearCanvas: function() {
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);
  }
};

var neural = {
  worker: new Worker("javascripts/neural-network.js"),
  initialize: function() {
    neural.worker.onmessage = neural.receiveMessage;
    neural.construct();
  },
  construct: function() {
    var message = JSON.stringify({command: "construct", data: config});
    neural.worker.postMessage(message);
    network.construct();
  },
  train: function() {
    var inputs = neural.parseInputs(document.getElementById("inputs").value.split("|"));
    var message = JSON.stringify({command: "train", data: inputs});
    neural.worker.postMessage(message);
  },
  validate: function() {
    var inputs = neural.parseInputs(document.getElementById("validation").value.split("|"));
    var message = JSON.stringify({command: "validate", data: inputs});
    neural.worker.postMessage(message);
  },
  parseInputs: function(inputs) {
    inputs.shift();
    for (var i = 0; i < inputs.length; i++) {
      var inputSegments = inputs[i].split(":");
      var input = {inputs: inputSegments[0].trim().split(" "), targets: inputSegments[1].trim().split(" ")};
      for (var j = 0; j < input.inputs.length; j++) {
        input.inputs[j] = parseFloat(input.inputs[j]);
      }
      for (var j = 0; j < input.targets.length; j++) {
        input.targets[j] = parseFloat(input.targets[j]);
      }
      inputs[i] = input;
    }
    return inputs;
  },
  receiveMessage: function(e) {
    var message = JSON.parse(e.data);
    if (message.hasOwnProperty("data")) {
      router[message.response](message.data);
    } else {
      router[message.response]();
    }
  }
};

var graphics = {
  line: function(start, end) {
    context.beginPath();
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
    return context;
  },
  circle: function(center, radius) {
    context.beginPath();
    context.arc(center.x, center.y, radius, 2 * Math.PI, false);
    return context;
  }
};

// neural.initialize();

var router = {
  log: function(message) {
    console.log(message);
  },
  constructed: function() {
    neural.train();
  },
  trained: function() {
    neural.validate();
  },
  validated: function() {

  },
  connect: function(data) {
    network.connect(data);
  },
  inputReceived: function(data) {
    network.inputReceived(data);
  },
  feedForward: function(data) {
    network.feedForward(data);
  },
  backPropagate: function(data) {
    network.backPropagate(data);
  },
  updateDelta: function(data) {
    network.updateDelta(data);
  }
};




// CLASS: Layer
function Layer(type) {
  this.type = type;
  this.perceptrons = [];

  this.connectToLayer = function(layer) {
    for (var i = 0; i < this.perceptrons.length; i++) {
      this.perceptrons[i].connectToLayer(layer);
    }
  }
  this.draw = function() {
    for (var i = 0; i < this.perceptrons.length; i++) {
      this.perceptrons[i].draw();
    }
  }
}

// CLASS: Perceptron
function Perceptron(x, y) {
  this.x = x;
  this.y = y;
  this.radius = 30;
  this.width = 8;
  this.color = "#000000";
  this.backgroundColor = "#ffffff";
  this.leftConnections = [];
  this.rightConnections = [];
  this.message = "";

  this.connectToLayer = function(layer) {
    for (var i = 0; i < layer.perceptrons.length; i++) {
      this.rightConnections.push(network.connections.length);
      layer.perceptrons[i].leftConnections.push(network.connections.length);
      network.connections.push(new Connection(this, layer.perceptrons[i]));
    }
  }
  this.inputReceived = function(message) {
    this.message = message;
    this.color = "yellow";
    this.backgroundColor = "yellow";
  }
  this.feedForward = function(message) {
    this.message = message;
    this.color = "#00ff00";
    this.backgroundColor = "#00ff00";
  }
  this.backPropagate = function() {
    this.color = "#ff0000";
  }
  this.updateDelta = function(message) {
    this.message = message;
    this.color = "#ff0000";
    this.backgroundColor = "#ff0000";
  }
  this.draw = function() {
    for (var i = 0; i < this.rightConnections.length; i++) {
      this.rightConnections[i].draw();
    }

    context.strokeStyle = this.color;
    context.fillStyle = this.backgroundColor;
    context.lineWidth = this.width;
    graphics.circle({x: this.x, y: this.y}, this.radius).stroke();
    graphics.circle({x: this.x, y: this.y}, this.radius).fill();

    context.fillStyle = "#000000";
    context.font = "bold 12px Tahoma";
    context.textAlign = "center";
    context.fillText(this.message, this.x, this.y + 4);
  }
}

// CLASS: Connection
function Connection(left, right, weight) {
  this.weight = weight;
  this.color = "#000000";
  this.left = left;
  this.right = right;
  this.width = 3;

  this.inputReceived = function() {
    this.color = "yellow";
  }
  this.feedForward = function() {
    this.color = "#00ff00";
  }
  this.backPropagate = function() {
    this.color = "#ff0000";
  }
  this.draw = function() {
    context.strokeStyle = this.color;
    context.lineWidth = this.width;
    graphics.line({x: this.left.x, y: this.left.y}, {x: this.right.x, y: this.right.y}).stroke();
  }
}
