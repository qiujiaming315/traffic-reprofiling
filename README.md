# Network Bandwidth Minimization

*You should have Python 3.7+, and an community edition of octeract installed to run this code.*

## Usage

The main script for running the minimization algorithm is `optimization.py`. Running them with `-h` gives you the various parameters you can
control:

```
usage: optimization.py [-h] net flow out

positional arguments:
  net                  Path to the input npy file describing network topology and flow routes.
  flow                 Path to the input npy file describing flow profiles.
  out                  Directory to save results.
```

- The input npy files for `net` and `flow` arguments can be generated in the following way:
    - In the "Input" directory, there is a `create_network.py` script.  You can either 
      specify your own network topology (the main function of that script gives an example),
      or call the `generate_random_net` function to create a random feed-forward network.
    - In the same directory you can also find a `create_flow.py` script, which is responsible
      for genereating flow profiles.  The usage is very similar to `create_network.py`.
      You can choose to either specify a profile or generate a random profile.

## Code

Various parts of the code are factored into different modules in the `lib/`
directory. You can begin by looking at the main training script to see how
these are called.

- `utils.py`: Utility functions for checking the format of input data.
- `network_parser.py`: Functions to do basic calculations for the given network,
  including parsing it from flow-node to flow-link format, and computing the
  rate-proportional solution for comparison.
- `genetic.py`: A genetic algorithm class with useful helper functions that implement
  details of the genetic algorithm.
- `order_generator.py`: Functions that propose random deadline orderings.
- `octeract.py`: Functions to translate the network and flow ordering into NLP, and call
  octeract solver to solve the generated NLP.
- `ampl.py`: Functions to formulate the network and flow ordering into AMPL files.
  If you prefer to use other solvers that take AMPL as one of the input formats,
  you can call functions in this script to generate AMPL files.
