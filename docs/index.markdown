---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---


Hello, This is a quick rundown of my master thesis [Exploration of the usage of semantic reasoning in reinforcement learning](https://github.com/wernerolaf/Semantic-Reasoning-in-Reinforcement-Learning) for more information click the link. It will redirect you to the project repository.

# Model Summary

In this master thesis, we tested if it is possible to train RL (Reinforcement Learning) algorithm using word embeddings and perform further reasoning on these embeddings. 

Here is a summary of the algorithm
1. Rescale and grayscale image
2. Find objects using Otsu thresholding
3. Compress objects with PCA and cluster them with Birch
4. Transform objects into a graph
5. Transform the graph into embedding with graph2vec
6. Use Reasoner to find the best path on State Map
7. Take action using Q-learning Neural Network using embedding and proposed the best path


# Starting State

We test architecture on the Pong game. We use the "PongNoFrameskip-v4" environment from OpenAI Gym.

<video src="assets/orig.ogg" type="video/ogg" width="320" height="240" controls="controls" style="max-width: 730px;">
</video>


# Basic Preprocessing

We rescale the image to 84x84 by using inter-area interpolation and grayscale.

<video src="assets/grey.ogg" type="video/ogg" width="320" height="240" controls="controls" style="max-width: 730px;">
</video>

# Object detection.

We detect objects using Otsu thresholding. This method separates objects into background and foreground. If pixels are connected then they are the same object. If an object is too big then we cut it.

<video src="assets/otsu.ogg" type="video/ogg" width="320" height="240" controls="controls" style="max-width: 730px;">
</video>

# Object clustering

After detecting objects we cluster them by first compressing them with PCA and then labeling them in an unsupervised manner with the Birch Clustering algorithm. Each color on the video represents a different object.

<video src="assets/birch.ogg" type="video/ogg" width="320" height="240" controls="controls" style="max-width: 730px;">
</video>

# Transforming to Graph Embedding

Now is the main part of the thesis. We transform our objects into nodes of the graph and connect them if they are close enough. However, We can not give our deep-Q network this graph directly. So we embed it. Embedding is transforming a graph into a vector. We do it using Graph2Vec. In short, we transform nodes into words/tokens and use Weisfeiler Lehman Hashing to encode their surroundings. We also add spatial information about each Node. Then we use the Gensim implementation of Doc2Vec which transforms word paragraphs into vectors. 

## Example node after embedding

```

'8', 'pos\_x4', 'pos\_y21', 'apr\_pos\_x0', 'apr\_pos\_y4', '8pos\_x4', '8pos\_y21', '8apr\_pos\_x0', '8apr\_pos\_y4', '8apr\_pos\_x0apr\_pos\_y4', '8pos\_x48pos\_y21', 'apr\_pos\_x0apr\_pos\_y4', 'pos\_x48pos\_y21',
'dab5edacccee84e7f5d123489f3e151c','4975d4c3aeda67b432297ab45ea7e3f8'

```

<video src="assets/graph.ogg" type="video/ogg" width="320" height="240" controls="controls" style="max-width: 730px;">
</video>


# Reasoner

## Elements
For our ontology and reasoning, we will be using the graph. Our graph holds information regarding state dynamics. It is made out of the following elements:
1. Node birch tree for storing nodes
2. Edge birch tree for storing edges
3. State prediction neural network
4. Reward prediction SVM

The node tree is clustering embeddings and these clusters are nodes.
Edge tree is clustering transitions between embeddings and these clusters are edges.
We create a graph by classifying the start and end embeddings of edge tree clusters using a node tree.
Because centroids are not "real" states, edges in reality can take several actions to cross. Therefore we also train models for predicting the next state and reward given the current state, the difference between the current state and the previous state, and chosen action.

## Long-term reasoning

We want to find the path with the greatest reward. This problem can be seen as the shortest path problem with weighted edges. Positive rewards are negative weights and vice versa. The existence of negative weights introduces a few problems. Dijkstra algorithm for finding shortest paths does not work when we have negative edges. Other algorithms like Bellman-Ford can work with negative edges, however, they still fail when there are negative cycles in the graph. To handle these issues, we will be searching for the closest positive reward while avoiding negative rewards using Djikstra algorithm with cutoff.

## Short-term reasoning

When we can predict both states and rewards we can implement short-term planning. It works like this: from starting state generate all possible action sequences with a given depth. Depth is the length of these sequences. Then for each sequence predict the end state and total reward, using predictions of the previous states.

## Summary of reasoning

The complete reasoning procedure is as follows:
1. Assign cluster to current state using node clustering Birch
tree.
2. Use the Dijkstra algorithm to find the shortest path to the
state which has a positive reward edge adjacent to it.
3. Generate scenarios using state and reward prediction models with chosen depth.
4. Sort scenarios with the following priorities: reward, the furthest node on the proposed path, and euclidean distance to the final node centroid.
5. For grounding, return information about the best scenario. This information is the proposed action sequence, predicted state, and reward while following this sequence.


The video below represents State Graph. Node colors represent the following things:
- yellow: neutral state
- red: negative state
- green: positive reward
- blue: current state
- orange: final state proposed by the reasoner

Node positions are based on the first two principal components of PCA of centroids of embeddings (which have 128 dimensions). Components explain only 10.8% of the variance which may be the reason why red and green nodes are not easily separable. After finishing training our node Birch tree got 2184 centroids and our edge Birch tree got 6969 centroids. If we try to classify states and transitions of our 3875 timesteps long test run we get 960 unique node labels and 1213 unique edge labels. So we compressed states by a factor of 4 and transitions by a factor of around 3. The diameter of this graph is 20 and the radius is 9.
 
<video src="assets/semantic.ogg" type="video/ogg" width="320" height="240" controls="controls" style="max-width: 730px;">
</video>


# Results

In the end, we managed to train the model based on semantic embedding which can play Pong. We also used reasoning to generate subgoals for our agents. Although the prediction of both states and rewards was far from perfect it still helped. The greatest obstacle to this thesis which was not directly overcome but rather avoided was the fact that Gensim implementation of doc2vec DOES NOT support incremental learning (see github issues [issue1](https://github.com/RaRe-Technologies/gensim/issues/1019), [issue2](https://github.com/RaRe-Technologies/gensim/issues/3162)). In future works, graph embedding will be further explored as graphs can represent any data structure and embeddings can make them work with many different algorithms. For more details go to [Exploration of the usage of semantic reasoning in reinforcement learning](https://github.com/wernerolaf/Semantic-Reasoning-in-Reinforcement-Learning).

# Used Technology

Here is a summary of the used technology
- Python: Programming language
- torch: Deep-Q learning
- gym: Pong environment
- sklearn: PCA and Birch
- skimage: image processing, Otsu thresholding
- gensim: doc2vec
- networkx: graphs, Djikstra

