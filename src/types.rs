use std::fs::File;

type ActivationFn = dyn Fn(&NeuralNetwork, f64) -> f64;

pub struct Counts
{
    pub input_neurons:                      u64,
    pub hidden_layers:                      u64,
    pub hidden_layer_neurons:               u64,
    pub output_neurons:                     u64,
}

pub struct Totals
{
    pub weights:                            u64,
    pub neurons:                            u64,
}

pub struct Activations
{
    pub hidden:                             Box<ActivationFn>,
    pub output:                             Box<ActivationFn>,
}

pub struct NeuralNetwork
{
    pub counts:                             Counts,
    pub totals:                             Totals,
    pub activations:                        Activations,
    pub interval:                           f64,
    pub weight:                             Vec<f64>,
    pub output:                             Vec<f64>,
    pub delta:                              Vec<f64>,
    pub lookup:                             Vec<f64>
}

pub trait NeuralNetworkLogic
{
    fn activation_hidden(&self, a: f64) ->  f64;
    fn activation_output(&self, a: f64) ->  f64;
    fn sigmoid(&self, a: f64) ->            f64;
    fn init_sigmoid_lookup(&mut self) ->    ();
    fn sigmoid_cached(&self, a: f64) ->     f64;
    fn linear(&self, a: f64) ->             f64;
    fn threshold(&self, a: f64) ->          f64;
    fn init(&mut self, counts: Counts) ->   ();
    fn read(&self, file: File) ->           NeuralNetwork;
    fn randomise(&self) ->                  ();
    fn copy(&self) ->                       NeuralNetwork;
    fn run(&self, inputs: Vec<f64>) ->      Vec<f64>;
    fn train
    (
        &self,
        inputs: Vec<f64>,
        expected_outputs: Vec<f64>,
        learning_rate: f64
    ) ->                                    ();
    fn write(&self, out: File) ->           ();
}
