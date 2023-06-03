# DL-Approx-Stochastic-Control

This repository contains the project work for the course "Computational Methods for Stochastic Differential Equations and Machine Learning" at the Royal Institute of Technology (KTH) in the spring term 2023, a joint work with Wille Viman.
The purpose of the project was to create a poster based on the paper ["Deep Learning Approximation for Stochastic Control Problems"](https://arxiv.org/pdf/1611.07422.pdf) by Han et al.

The paper proposes a method to approximate optimal controllers for stochastic control setups using stacked deep neural networks.
The network consists of a subnetwork for each time step (i.e. $T$ subnetworks for a control problem with finite horizon $T$), where each subnetwork predicts an action for a given state, which is then used to update the state and provide it as an input for the next subnetwork.
This network architecture was implemented in Python using TensorFlow2 and tested for a simple toy example, a model of a warehouse with stochastic demand.
The poster as shown below summarizes the most important aspects of the approach as well as then obtained results.

<img width="1390" alt="poster" src="https://github.com/jonahaag/DL-Approx-Stochastic-Control/assets/44704480/75c976f2-ea2e-45dc-a7cc-77673a105754">
