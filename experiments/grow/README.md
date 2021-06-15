# Growing Virtual Creatures in a World Without Physics

## Why a World Without Physics?

The primary reason is practical. Right now, it's not possible to make meaningful progress exploring the efficiacy of AI in a true physics simulation on a desktop computer. Although academic clusters and cloud services are available, they tend to be cumbersome in the absence of root access (and hence productivity tools like Docker, although this is [changing]()) and expensive on which to iterate respectively. I run an AI group and support a family; letting my desktop hum takes much less time and is less expensive. 

A really nice consequence of exploring virtual creatures in environments that are computationally feasible for me is that they are likely computationally feasible for you, too. Many of us have valuable expertise we apply daily on our teams; providing a platform with which to explore moonshot style research is a major goal of this project. If I run an experiment on monitor 2 while playing PUBG on monitor 1, so can you.

Practicallity aside, there are plenty of good reasons to consider growing creatures outside of a simulation that approximates the laws of the universe we know. This universe is not the only universe we can observe and learn from. Reinforcement learning has benefited a great deal from Atari [], for example. It's not unlikely that what we learn in a simple gridworld, or a gridworld with some simple rules can translate back to our own and other universes. I'm also really interested in how creatures might transfer from one universe to the next; optimizing for our universe is only useful if you want to build robots that can walk around and make coffee.

## Theory

There are two ways to describe this work. One is with symbols and depends on a broad knowledge of the field and the other is with verbose descriptions and analogies. This README will address the former. For the later checkout [this Medium post]().

### Motivation 

This research is about growing instead of engineering. The previous decade has seen incredible strides in machine learning, notably in areas such as image recognition and natural language process. Many of the advancements in these areas have depended on designing domain specific techniques and neural architectures that smooth out the search landscape during function approximation. Although the resulting models have been performant, careful research, experimentation, and dissemination of results is at best a process very slow to scale and at worst unscalable. Apropos, Blei et al. [] investigate the feasibility of inferring the posterior for generic probabilistic models, something we take for granted with the widespread success of Stochastic Gradient Descent.

Machine learning is simple function approximation over a constant data set that seeks to minimize empirical risk []; pattern recognition as popularized by the title of Bishop's textbook []. Indeed we can view functions found in machine learning as components of a larger function that govern the decision making processs in reinforcement learning. For example when the state space is a two-dimensional screen the first several layers of the policy function are convoultional. Of course this simple decision making in a heavily constrained environment with a static morphology: with a more dynamic morphology and environment one would expect a variety of input organizing pattern recognizers to be layered above transformations capable of abstracting input into ideas and decisions. Designing creature and environment sepcific sensor architectures and the lower levels of abstraction by hand, paper by paper, will not scale: engineering piece by piece is a dead end []. To that end, we explore the possibility of farming and growing creatures rather than designing them.

Here's the idea: we define a function that grows a creature and we optimize that function to grow creatures we like. In the experiments on this page we'll be looking at three dimensional geometries that do things like maximize surface area. These initial experiments are designed to probe the hypothesis that a function can be trained to instruct the growth of creatures that maximize a reward function. Although we explore these creature in a three dimensional, psudo-physical space, a creature can be anything that can grow iteratively: a document, a neural network, a microchip, anything. 

### Construction

Consider the conditional distribution that 
