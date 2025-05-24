example is a bioprocess model for a fermentation process with algebraic kinetics, 
states: biomass, glucose, acetate, product
inputs: glucose feed
measurements: biomass, acetate

kinetic model: monod model with double substrate inhibition for growth, haldane model for product formation

- first create an example dataset csv file with the following columns: processID, time, glucose feed, biomass, acetate, product
Simulate three different processes with different glucose feed profiles and parameter values

- then create a model with the following states: biomass, glucose, acetate, product