use crate::types::
{
    NeuralNetwork,
    NeuralNetworkLogic,
    Counts
};

const LOOKUP_SIZE:          usize = 4096;
const SIGMOID_DOM_MIN:      f64 = -15.0;
const SIGMOID_DOM_MAX:      f64 = 15.0;
const F: f64 = (SIGMOID_DOM_MAX - SIGMOID_DOM_MIN) / LOOKUP_SIZE as f64;

impl /*NeuralNetworkLogic for*/ NeuralNetwork
{
    fn activation_hidden(&self, a: f64) -> f64
    {
        let f = &self.activations.hidden;
        return f(self, a);
    }

    fn activation_output(&self, a: f64) -> f64
    {
        let f = &self.activations.hidden;
        return f(self, a);
    }

    fn sigmoid(&self, a: f64) ->            f64
    {
        if a < -45.
        {
            return 0.;
        }
        if a > 45.
        {
            return 1.;
        }
        return 1. / (1. + (-a as f64).exp());
    }

    fn sigmoid_lookup(&mut self)
    {
        let i: usize;

        i = 0;
        self.interval = LOOKUP_SIZE as f64 / (SIGMOID_DOM_MAX - SIGMOID_DOM_MIN);

        while i < LOOKUP_SIZE
        {
            self.lookup[i] = self.sigmoid(SIGMOID_DOM_MIN + F * i as f64);
            i += 1;
        }
    }

    fn sigmoid_cached(&self, a: f64) -> f64
    {
        assert!(!a.is_nan());
        let j: usize;

        if a < SIGMOID_DOM_MIN
        {
            return self.lookup[0];
        }
        if a >= SIGMOID_DOM_MAX
        {
            return self.lookup[LOOKUP_SIZE - 1];
        }
        j = ((a - SIGMOID_DOM_MIN) * self.interval + 0.5) as usize;
        if j >= LOOKUP_SIZE
        {
            return self.lookup[LOOKUP_SIZE - 1];
        }
        return self.lookup[j];
    }

    fn linear(&self, a: f64) -> f64
    {
        return a;
    }

    fn threshold(&self, a: f64) -> f64
    {
        return if a > 0. { 1. } else { 0. };
    }

    fn init(&mut self, counts: Counts) -> ()
    {
        let mut hidden_weights: u64;
        let mut output_weights: u64;

        assert!(counts.input_neurons >= 1);
        assert!(counts.output_neurons >= 1);
        assert!(!(counts.hidden_layers < 1 && counts.hidden_layer_neurons >= 1));
        if counts.hidden_layers > 0
        {
            hidden_weights = (counts.input_neurons + 1) * counts.hidden_layer_neurons
            +
            (counts.hidden_layers - 1) * (counts.hidden_layer_neurons + 1) * counts.hidden_layer_neurons;
        } else {
            hidden_weights = 0;
        }
        if counts.hidden_layers > 0
        {
            output_weights = counts.hidden_layer_neurons + 1;
        } else {
            output_weights = (counts.input_neurons + 1) * counts.output_neurons;
        }
        self.totals.weights = hidden_weights + output_weights;
        self.totals.neurons = counts.input_neurons + counts.hidden_layer_neurons * counts.hidden_layers + counts.output_neurons;
        self.counts = counts;
        self.weight = Vec::new();
        self.output = Vec::new();
        self.delta = Vec::new();
        self.randomise();
        self.activations.hidden = self.sigmoid_cached;
        self.activations.output = self.sigmoid_cached;
        self.sigmoid_lookup();
    }
}
