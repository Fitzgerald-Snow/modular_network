"""
Krotov-Hopfield Dense Associative Memory implementation for continuous patterns.
WITHOUT the initialize function - relies on SNMF's automatic cluster discovery.

Based on: "Large Associative Memory Problem in Neurobiology and Machine Learning"
by Krotov & Hopfield (2021), arXiv:2008.06996

This version tests whether SNMF can automatically discover the clustering pattern
without explicit supervised initialization, as suggested by Pehlevan & Chklovskii (2014).
"""

import jax.numpy as jnp
import jax
from jax.scipy.special import logsumexp
import numpy as np
from scipy import optimize


class SNMF_Continuous():
    """
    Online clustering by Symmetric Non-negative Matrix Factorization Algorithm,
    adapted for continuous patterns.

    Instead of XOR for binary patterns, we use:
    - Hidden = Input - Reference (storage)
    - Reconstructed = Hidden + Reference (retrieval)
    """
    def __init__(self, n_input_neurons, n_gate_neurons, decay1=0.99, decay2=0.99) -> None:
        """
        decay1: used to decay Y;
        decay2: used to decay the weights from gate neurons to hidden layer (reference points).
        """
        self.n_input_neurons = n_input_neurons
        self.n_gate_neurons = n_gate_neurons
        self.decay1, self.decay2 = decay1, decay2
        self.W = np.random.normal(scale=0.01, size=(n_gate_neurons, n_input_neurons))  # Hebbian weights
        self.M = np.random.normal(scale=0.01, size=(n_gate_neurons, n_gate_neurons))  # Anti-Hebbian weights
        np.fill_diagonal(self.M, 0.)
        self.y = None  # activity of gate neurons
        self.Y = np.ones((n_gate_neurons, )) * 100
        # Reference points for each gate neuron (replaces XOR operation)
        self.reference_points = np.zeros((n_input_neurons, n_gate_neurons))
        self.active_y_idx = 0

    def get_gate_layer_output(self, x, state_changeQ=True, n_iterations=10) -> None:
        """
        Calculate self.y, self.active_y_idx.
        If state_changeQ is True, self.Y will also be changed.
        """
        w_mul_x = self.W @ x

        def fun(x2):
            return x2 - np.maximum(w_mul_x - self.M @ x2, 0)

        y = optimize.fsolve(fun, w_mul_x)
        self.y = y

        if state_changeQ is True:
            self.Y = self.decay1 * self.Y + y ** 2
        self.active_y_idx = np.argmax(y)

    def get_hidden_layer_output(self, x) -> np.ndarray:
        """
        Calculate the hidden layer pattern by subtracting the reference point.
        For continuous patterns: hidden = input - reference
        """
        return x - self.reference_points[:, self.active_y_idx]

    def reconstruct_input(self, hidden) -> np.ndarray:
        """
        Reconstruct the input pattern by adding back the reference point.
        For continuous patterns: reconstructed = hidden + reference
        """
        return hidden + self.reference_points[:, self.active_y_idx]

    def update_weights(self, x) -> None:
        """Update self.W, self.M, and reference_points"""
        tmp1 = (self.y / self.Y).reshape((-1, 1))
        tmp2 = (self.y ** 2 / self.Y).reshape((-1, 1))
        self.W += tmp1 @ (x.reshape((1, -1))) - tmp2 * self.W
        self.M += tmp1 @ (self.y.reshape((1, -1))) - tmp2 * self.M
        np.fill_diagonal(self.M, 0.)
        # Update reference point with exponential moving average
        self.reference_points[:, self.active_y_idx] = (
            self.decay2 * self.reference_points[:, self.active_y_idx] +
            x * (1 - self.decay2)
        )


class KrotovHopfield():
    """
    Dense Associative Memory (Modern Hopfield Network) from Krotov & Hopfield (2021).

    Uses softmax update rule for pattern retrieval:
        x_new = Ξ^T softmax(β * Ξ x)

    where Ξ is the matrix of stored patterns and β is inverse temperature.

    This replaces the classical Hopfield network and supports continuous patterns.
    """
    def __init__(self, n_neurons, n_clusters, beta=1.0, decay=0.99, max_patterns_per_cluster=100) -> None:
        """
        n_neurons: dimension of each pattern
        n_clusters: number of separate associative memory modules
        beta: inverse temperature (higher = sharper softmax, more like winner-take-all)
        decay: decay rate for online learning
        max_patterns_per_cluster: maximum number of patterns that can be stored per cluster
        """
        self.n_neurons = n_neurons
        self.n_clusters = n_clusters
        self.beta = beta
        self.decay = decay
        self.max_patterns = max_patterns_per_cluster

        # Store patterns explicitly for each cluster
        # Shape: (n_clusters, max_patterns, n_neurons)
        self.patterns = np.zeros((n_clusters, max_patterns_per_cluster, n_neurons))
        # Number of patterns stored in each cluster
        self.pattern_counts = np.zeros(n_clusters, dtype=int)
        # Running average for pattern normalization (optional)
        self.pattern_norms = np.ones((n_clusters, max_patterns_per_cluster))

    def weight_update(self, x, active_cluster_idx) -> None:
        """
        Store a new pattern in the specified cluster.
        Uses exponential moving average for online learning.
        """
        cluster_idx = active_cluster_idx
        count = self.pattern_counts[cluster_idx]

        # the number of patterns stored in this cluster cannot exceed the total number of neurons in hidden layer
        if count < self.max_patterns:
            # Add new pattern
            # the synaptic weight is then simply the pattern itself
            self.patterns[cluster_idx, count] = x
            self.pattern_counts[cluster_idx] += 1
        else:
            # Online learning: update existing patterns with decay
            # Find the pattern most similar to x and update it
            similarities = self.patterns[cluster_idx, :count] @ x
            most_similar_idx = np.argmax(similarities)
            self.patterns[cluster_idx, most_similar_idx] = (
                self.decay * self.patterns[cluster_idx, most_similar_idx] +
                (1 - self.decay) * x
            )

    def retrieve(self, x, active_cluster_idx, n_iterations=10) -> np.ndarray:
        """
        Retrieve a pattern using the softmax update rule (Modern Hopfield dynamics).

        Update rule: x_new = Ξ^T softmax(β * Ξ x)
        """
        cluster_idx = active_cluster_idx
        count = self.pattern_counts[cluster_idx]

        if count == 0:
            return x  # No patterns stored, return input

        # Get stored patterns for this cluster
        Xi = self.patterns[cluster_idx, :count]  # Shape: (count, n_neurons)

        for _ in range(n_iterations):
            # Compute similarities: Ξ x
            similarities = Xi @ x  # Shape: (count,)

            # Softmax with inverse temperature β
            # Using logsumexp for numerical stability
            log_weights = self.beta * similarities
            log_normalizer = logsumexp(log_weights)
            weights = np.exp(log_weights - log_normalizer)  # Shape: (count,)

            # Update: x_new = Ξ^T softmax(β * Ξ x)
            x = Xi.T @ weights  # Shape: (n_neurons,)

        return x


class TreeNetworkKrotovNoInit():
    """
    A tree-like network for hierarchical memory classification, storage, and retrieval.

    This version does NOT have an initialize function - it relies entirely on
    SNMF's ability to automatically discover cluster structure through online learning.

    Uses:
    - SNMF_Continuous for clustering (with reference point subtraction/addition)
    - KrotovHopfield for pattern storage and retrieval (softmax dynamics)
    """
    def __init__(self, n_input_neurons, tree_struct_list, tree_depth=0,
                 tree_decay1_list=(0.99,), tree_decay2_list=(0.99,),
                 decay3=0.99, beta=1.0, max_patterns=100) -> None:
        """
        tree_struct_list: tuple (n1, n2, ...) defining the tree structure
        tree_depth: depth of this node in the tree
        tree_decay1_list: decay for Y in SNMF modules
        tree_decay2_list: decay for reference points in SNMF modules
        decay3: decay for Krotov-Hopfield networks
        beta: inverse temperature for softmax in Krotov-Hopfield
        max_patterns: maximum patterns per cluster in Krotov-Hopfield
        """
        self.tree_depth = tree_depth
        self.n_gate_units = tree_struct_list[0]
        decay1 = tree_decay1_list[0]
        decay2 = tree_decay2_list[0]

        self.snmf = SNMF_Continuous(n_input_neurons, self.n_gate_units, decay1, decay2)
        self.next_tree_struct_list = tree_struct_list[1:]
        self.hidden = np.zeros(n_input_neurons)

        if len(tree_struct_list) > 1:
            self.sub_trees = [
                TreeNetworkKrotovNoInit(
                    n_input_neurons, self.next_tree_struct_list,
                    tree_depth=self.tree_depth + 1,
                    tree_decay1_list=tree_decay1_list[1:],
                    tree_decay2_list=tree_decay2_list[1:],
                    decay3=decay3, beta=beta, max_patterns=max_patterns
                ) for _ in range(self.n_gate_units)
            ]
        else:
            self.sub_trees = []
            self.hopfields = KrotovHopfield(
                n_input_neurons, n_clusters=self.n_gate_units,
                beta=beta, decay=decay3, max_patterns_per_cluster=max_patterns
            )

        self.count_record = np.zeros(self.n_gate_units)

    def __repr__(self) -> str:
        rep = f"depth: {self.tree_depth}, gate units: {self.n_gate_units}.\n"
        if self.sub_trees == []:
            rep += f"Leaf layer with {self.n_gate_units} Krotov-Hopfield networks\n"
        else:
            rep += f"Each gate controls subtree with structure {self.next_tree_struct_list}.\n"
        return rep

    def update_weights(self, x, trainSNMFQ=True, trainHopfieldQ=True) -> None:
        """Update weights throughout the tree."""
        if trainSNMFQ:
            self.snmf.get_gate_layer_output(x, state_changeQ=True)
            self.snmf.update_weights(x)
        else:
            self.snmf.get_gate_layer_output(x, state_changeQ=False)

        # Get hidden representation (input - reference)
        y = self.snmf.get_hidden_layer_output(x)

        if self.sub_trees == []:
            if trainHopfieldQ:
                self.hopfields.weight_update(y, self.snmf.active_y_idx)
        else:
            self.sub_trees[self.snmf.active_y_idx].update_weights(y, trainSNMFQ, trainHopfieldQ)

    def forward_classify(self, x, state_changeQ=False, countingQ=False, n_hopfield_iterations=0):
        """
        Forward pass: classify x and optionally retrieve using Krotov-Hopfield.
        """
        self.snmf.get_gate_layer_output(x, state_changeQ=state_changeQ)
        self.hidden = self.snmf.get_hidden_layer_output(x)
        active_gate_unit_idx = self.snmf.active_y_idx

        if self.sub_trees == []:
            classification_result = [active_gate_unit_idx]
            residual = self.hidden
            residual = self.hopfields.retrieve(
                residual, active_cluster_idx=active_gate_unit_idx,
                n_iterations=n_hopfield_iterations
            )
        else:
            tmp = self.sub_trees[active_gate_unit_idx].forward_classify(
                self.hidden, state_changeQ, countingQ, n_hopfield_iterations
            )
            classification_result = [active_gate_unit_idx] + tmp[0]
            residual = tmp[1]

        if countingQ:
            self.count_record[active_gate_unit_idx] += 1

        return (classification_result, residual)

    def backward_reconstruction(self, x) -> np.ndarray:
        """Recursive backward reconstruction by adding back reference points."""
        if self.sub_trees != []:
            x = self.sub_trees[self.snmf.active_y_idx].backward_reconstruction(x)
        return self.snmf.reconstruct_input(x)

    def retrieve(self, x, n_hopfield_iterations, countingQ=False) -> np.ndarray:
        """Retrieve a pattern (possibly noisy) from memory."""
        _, residual = self.forward_classify(
            x, state_changeQ=False, countingQ=countingQ,
            n_hopfield_iterations=n_hopfield_iterations
        )
        return self.backward_reconstruction(residual)

    # NOTE: initialize function is intentionally REMOVED
    # The SNMF algorithm should discover clusters automatically through online learning

    def get_misclassification_percentage(self) -> float:
        avg = np.mean(self.count_record)
        if np.sum(self.count_record) == 0:
            return 0.0
        return np.sum(np.abs(self.count_record - avg)) / 2 / np.sum(self.count_record)

    def reset_counting_record(self) -> None:
        self.count_record = np.zeros(self.n_gate_units)
        if self.sub_trees != []:
            for idx in range(self.n_gate_units):
                self.sub_trees[idx].reset_counting_record()

    def reset_hopfield_weights(self) -> None:
        if self.sub_trees != []:
            for idx in range(self.n_gate_units):
                self.sub_trees[idx].reset_hopfield_weights()
        else:
            self.hopfields.patterns = np.zeros(self.hopfields.patterns.shape)
            self.hopfields.pattern_counts = np.zeros(self.hopfields.n_clusters, dtype=int)


class UltraMetricTreeContinuous():
    """
    Generates hierarchical continuous patterns organized in an ultrametric tree.

    Children are created by adding isotropically chosen random unit vectors
    scaled by a fixed norm. The norm halves at each level: 1, 0.5, 0.25, ...
    """
    def __init__(self, key, n_neurons, tree_struct_list, base_norm=1.0):
        """
        tree_struct_list: (n1, n2, ...) structure of the tree
        base_norm: norm of displacement vectors at the first level (default 1.0)
                   subsequent levels use base_norm/2, base_norm/4, ...
        """
        self.key, subkey = jax.random.split(key)
        self.tree_struct_list = tree_struct_list
        self.base_norm = base_norm
        self.n_neurons = n_neurons
        # Root ancestor is the zero vector (serves as reference)
        self.root_ancestor = jnp.zeros((n_neurons,))
        self.descendents = [self.root_ancestor]
        self.keys = [subkey]

    def get_direct_descendents(self, ancestor, key, n_patterns_per_cluster, fixed_norm):
        """
        Generate descendants by adding isotropic random vectors of fixed norm.

        For each child:
        1. Sample a random vector from standard normal distribution
        2. Normalize to unit vector
        3. Scale by fixed_norm
        4. Add to ancestor

        descendant = ancestor + fixed_norm * (random_vector / ||random_vector||)
        """
        n_neurons = len(ancestor)

        # Generate random vectors from standard normal (isotropic direction)
        random_vectors = jax.random.normal(key, (n_patterns_per_cluster, n_neurons))

        # Normalize each vector to unit length, then scale by fixed_norm
        norms = jnp.linalg.norm(random_vectors, axis=1, keepdims=True)
        unit_vectors = random_vectors / norms
        displacement_vectors = unit_vectors * fixed_norm

        return ancestor + displacement_vectors

    def get_next_tree(self, pattern_tree, key_tree, n_patterns_per_cluster, fixed_norm):
        def fun(key, ancestor):
            descendents = self.get_direct_descendents(ancestor, key, n_patterns_per_cluster, fixed_norm)
            return [descendents[i] for i in range(len(descendents))]

        self.descendents = jax.tree.map(fun, self.keys, self.descendents)
        self.keys = jax.tree.map(
            lambda x: list(jax.random.split(x, n_patterns_per_cluster)),
            self.keys
        )

    def construct_tree(self):
        """Construct the full tree of continuous patterns."""
        for i in range(len(self.tree_struct_list)):
            n_patterns_per_cluster = self.tree_struct_list[i]
            # Norm halves at each level: base_norm, base_norm/2, base_norm/4, ...
            fixed_norm = self.base_norm / (2 ** i)
            self.get_next_tree(self.descendents, self.keys, n_patterns_per_cluster, fixed_norm)


# Utility functions for working with continuous patterns

def add_gaussian_noise(pattern, key, noise_std=0.1):
    """Add Gaussian noise to a continuous pattern."""
    noise = jax.random.normal(key, pattern.shape) * noise_std
    return pattern + noise


def compute_correlation(pattern1, pattern2):
    """Compute normalized correlation between two continuous patterns."""
    norm1 = jnp.linalg.norm(pattern1)
    norm2 = jnp.linalg.norm(pattern2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return jnp.dot(pattern1, pattern2) / (norm1 * norm2)


def compute_mse(pattern1, pattern2):
    """Compute mean squared error between two patterns."""
    return jnp.mean((pattern1 - pattern2) ** 2)
