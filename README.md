# Infinity Mirror experiments TODOs
0. Get the different synthetic graphs going on
pairwise graph distances, pick best, worst, and median.
ranked choice voting.. Instant run-off voting


1. Get graph generators working
    * **works** ER, CNRG, HRG, BTER, Kronecker, Chung-Lu
    * ForestFire, Chungu-Lu + clustering, Chung-Lu + transitivity, ERGM
    * Neural network based methods
        - Learning deep generative models for graphs, ICML 2018
        - GraphRNN - Leskovec
        - GraphVAE - ICANN
        - Constrained generation of semantically valid graphs - NeurIPS
        - MolGAN - de Cao, Kipf
        - NetGAN - Bojchevski
        - Graphite
        - other neural network based methods
    
2. Get graph similarity measures working
    * GCD
    * DeltaCon
    * CVM / KS test for distributions
    * Graphlet counts using PGD

3. Distributions
    * Degree
    * PageRank
    * Assortativity
    * Clustering coefficient
    * Hop Plots
    * Scree plot
    * Core decomposition
    
4. Matplotlib / webweb animations for visualizing generations of graphs
      
