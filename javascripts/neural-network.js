var neural;
var interval = 50;
var showLogs = false;
var showStatus = true;

onmessage = function(e) {
  var message = JSON.parse(e.data);
  if (message.hasOwnProperty("data")) {
    router[message.command](message.data);
  } else {
    router[message.command]();
  }
}

var router = {
  construct: function(config) {
    log("Constructing neural network...");
    neural = new NeuralNetwork(config);
    log("Neural network constructed.");
    send({response: "constructed"});
  },
  train: function(inputs) {
    log("Training neural network...");
    neural.train(inputs);
    log("Neural network trained.");
    send({response: "trained"});
  },
  validate: function(inputs) {
    log("Validating neural network...");
    neural.validate(inputs);
    log("Neural network validated.");
    send({response: "validated"});
  }
};

function log(message) {
  if (showLogs) {
    send({response: "log", data: message});
  }
}

function status(status, data) {
  if (showStatus) {
    send({response: status, data: data});
  }
}

function send(message) {
  postMessage(JSON.stringify(message));
}



// CLASS: NeuralNetwork
function NeuralNetwork(config) {
  this.eta = config.eta;
  this.momentum = config.momentum;
  this.min_mse = config.min_mse;
  this.max_epochs = config.max_epochs;
  this.layers = [];

  this.construct = function() {
    var structure = config.structure.split(" ");
    for (var i = 0; i < structure.length; i++) {
      var layerSize = parseInt(structure[i]);
      var type = "HIDDEN";
      if (i == 0) {
        type = "INPUT";
      } else if (i == structure.length - 1) {
        type = "OUTPUT";
      }

      this.layers.push(new Layer(type, layerSize, i));
    }

    log("  Connecting layers...");
    for (var i = 0; i < this.layers.length - 1; i++) {
      log("    Connecting " + this.layers[i].type + " layer to " + this.layers[i + 1].type + " layer...");
      this.layers[i].connectToLayer(this.layers[i + 1]);
    }
    log("  All layers connected.");
  }
  this.construct();

  this.train = function(data) {
    var epochs = 0;
    var mse = 1;
    var outputSummation = 0;
    var corrects = 0;

    var index = 0;
    while (epochs < this.max_epochs && mse > this.min_mse) {
      var inputs = [1].concat(data[index].inputs);
      var targets = data[index].targets;

      var outputs = this.feedForward(inputs);
      this.backPropagate(targets);

      outputSummation += this.calculateOutputSummation(targets, outputs);

      if (index == 0) {
        corrects = 0;
      }
      if (this.correctOutput(targets, outputs)) {
        corrects++;
      }

      index++;
      if (index == data.length) {
        index = 0;
        epochs++;
        mse = outputSummation / 2;
        outputSummation = 0;
        log("  " + epochs + ": " + mse);
      }
    }
  }
  this.validate = function(data) {
    var corrects = 0;
    for (var i = 0; i < data.length; i++) {
      var inputs = [1].concat(data[i].inputs);
      var targets = data[i].targets;

      outputs = this.feedForward(inputs);
      if (this.correctOutput(targets, outputs)) {
        corrects++;
      }
    }
    log("  Performance: " + corrects + "/" + data.length + " (" + (corrects / data.length * 100) + "%)");
  }
  this.feedForward = function(inputs) {
    for (var i = 0; i < this.layers.length; i++) {
      inputs = this.layers[i].feedForward(inputs);
    }
    return inputs;
  }
  this.backPropagate = function(targets) {
    for (var i = this.layers.length - 1; i >= 0; i--) {
      this.layers[i].backPropagate(targets, this.eta, this.momentum);
    }
  }
  this.calculateOutputSummation = function(targets, outputs) {
    var summation = 0;
    for (var i = 0; i < targets.length; i++) {
      summation += Math.pow(targets[i] - outputs[i], 2);
    }
    return summation;
  }
  this.correctOutput = function(targets, outputs) {
    for (var i = 0; i < targets.length; i++) {
      if (targets[i] != this.roundOutput(outputs[i])) {
        return false;
      }
    }
    return true;
  }
  this.roundOutput = function(output) {
    if (output >= 0.85) {
      return 1;
    } else if (output <= 0.15) {
      return 0;
    }
    return output;
  }
}

// CLASS: Layer
function Layer(type, size, index) {
  this.index = index;
  this.type = type;
  this.perceptrons = [];
  this.outputs = [];

  this.initialize = function() {
    size = (type == "INPUT") ? size + 1 : size;
    log("  Creating " + type + " layer... size: " + size);
    for (var i = 0; i < size; i++) {
      this.perceptrons.push(new Perceptron(this, i));
    }
  }
  this.initialize();

  this.connectToLayer = function(layer) {
    for (var i = 0; i < this.perceptrons.length; i++) {
      log("      Connecting " + type + " perceptron #" + i + " to " + layer.type + " layer...");
      this.perceptrons[i].connectToLayer(layer);
    }
  }
  this.feedForward = function(inputs) {
    for (var i = 0; i < this.perceptrons.length; i++) {
      var data = {
        layer: this.index,
        perceptron: this.perceptrons[i].index,
        value: inputs[i]
      };
      status("inputReceived", data);
      sleep(interval);
      this.outputs[i] = this.type == "INPUT" ? inputs[i] : this.perceptrons[i].calculateOutput(inputs);
      data.value = this.outputs[i];
      status("feedForward", data);
      sleep(interval);
    }
    return this.outputs;
  }
  this.backPropagate = function(targets, eta, momentum) {
    for (var i = 0; i < this.perceptrons.length; i++) {
      if (this.type == "OUTPUT") {
        this.perceptrons[i].calculateOutputNeuronDelta(targets[i], this.outputs[i]);
        var data = {
          layer: this.index,
          perceptron: this.perceptrons[i].index,
          value: this.perceptrons[i].delta
        };
        status("updateDelta", data);
        status("backPropagate", data);
        sleep(interval);
      } else if (this.type == "HIDDEN") {
        this.perceptrons[i].calculateHiddenNeuronDelta(this.outputs[i], i);
        this.perceptrons[i].updateWeights(eta, momentum, this.outputs[i], i);
        var data = {
          layer: this.index,
          perceptron: this.perceptrons[i].index,
          value: this.perceptrons[i].delta
        };
        status("updateDelta", data);
        status("backPropagate", data);
        sleep(interval);
      } else if (this.type == "INPUT") {
        this.perceptrons[i].updateWeights(eta, momentum, this.outputs[i], i);
      }
    }
  }
}

// CLASS: Perceptron
function Perceptron(layer, index) {
  this.index = index;
  this.layer = layer;
  this.leftConnections = [];
  this.rightConnections = [];
  this.leftWeights = [];
  this.deltaWeights = [];
  this.delta = 0;

  this.calculateOutput = function(inputs) {
    return sigmoid(vectorSum(inputs, this.leftWeights));
  }
  this.calculateOutputNeuronDelta = function(target, output) {
    this.delta = output * (1 - output) * (target - output);
  }
  this.calculateHiddenNeuronDelta = function(output, index) {
    var weights = this.getWeights(index);
    var deltas = this.getDeltas();
    this.delta = output * (1 - output) * vectorSum(weights, deltas);
  }
  this.getWeights = function(index) {
    var weights = [];
    for (var i = 0; i < this.rightConnections.length; i++) {
      weights.push(this.rightConnections[i].leftWeights[index]);
    }
    return weights;
  }
  this.getDeltas = function() {
    var deltas = [];
    for (var i = 0; i < this.rightConnections.length; i++) {
      deltas.push(this.rightConnections[i].delta);
    }
    return deltas;
  }
  this.updateWeights = function(eta, momentum, output, index) {
    for (var i = 0; i < this.rightConnections.length; i++) {
      var connection = this.rightConnections[i];
      connection.deltaWeights[index] = eta * connection.delta * output + momentum * connection.deltaWeights[index];
      connection.leftWeights[index] += connection.deltaWeights[index];
    }
  }
  this.connectToLayer = function(layer) {
    for (var i = 0; i < layer.perceptrons.length; i++) {
      this.connectRight(layer.perceptrons[i]);
      layer.perceptrons[i].connectLeft(this);
      var data = {
        leftLayer: this.layer.index, 
        leftPerceptron: this.index,
        rightLayer: layer.index,
        rightPerceptron: layer.perceptrons[i].index,
        weight: layer.perceptrons[i].leftWeights[layer.perceptrons[i].leftWeights.length - 1]
      }
      status("connect", data);
    }
  }
  this.connectLeft = function(perceptron) {
    this.leftConnections.push(perceptron);
    this.leftWeights.push(Math.random());
    this.deltaWeights.push(0);
  }
  this.connectRight = function(perceptron) {
    this.rightConnections.push(perceptron);
  }
}

// FUNCTIONS
function vectorSum(x, y) {
  var sum = 0;
  for (var i = 0; i < x.length; i++) {
    sum += x[i] * y[i];
  }
  return sum;
}

function sigmoid(x) {
  return 1 / (1 + Math.pow(Math.E, -x));
}

function sleep(ms) {
  if (showStatus) {
    var date1 = new Date();
    var date2 = new Date();
    while (date2.valueOf() < date1.valueOf() + ms) {
      date2 = new Date();
    }
  }
}