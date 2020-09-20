import numpy as np


class Mutations():

    def __init__(self, config):
        self.cfg = config
        self.rng = np.random.RandomState(self.cfg.searl.seed)

    def no_mutation(self, individual):
        individual.train_log["mutation"] = "no_mutation"
        return individual

    def mutation(self, population):

        mutation_options = [self.no_mutation, self.architecture_mutate, self.parameter_mutation, self.activation_mutation,
                            self.rl_hyperparam_mutation]
        mutation_choice = self.rng.choice(mutation_options, len(population))

        mutated_population = []
        for mutation, individual in zip(mutation_choice, population):
            mutated_population.append(mutation(individual))

        return mutated_population

    def rl_hyperparam_mutation(self, individual):

        rl_config = individual.rl_config
        rl_params = self.cfg.mutation.rl_hp_selection
        mutate_param = self.rng.choice(rl_params, 1)[0]

        random_num = self.rng.uniform(0, 1)
        if random_num > 0.5:
            setattr(rl_config, mutate_param, min(0.005, max(0.00001, getattr(rl_config, mutate_param) * 1.2)))
        else:
            setattr(rl_config, mutate_param, min(0.005, max(0.00001, getattr(rl_config, mutate_param) * 0.8)))

        individual.train_log["mutation"] = "rl_" + mutate_param
        individual.rl_config = rl_config
        return individual

    def activation_mutation(self, individual):
        individual.actor = self._permutate_activation(individual.actor)
        individual.critic_1 = self._permutate_activation(individual.critic_1)
        individual.critic_2 = self._permutate_activation(individual.critic_2)
        individual.train_log["mutation"] = "activation"
        return individual

    def _permutate_activation(self, network):

        possible_activations = ['relu', 'elu', 'tanh']
        current_activation = network.activation
        possible_activations.remove(current_activation)
        new_activation = self.rng.choice(possible_activations, size=1)[0]
        net_dict = network.init_dict
        net_dict['activation'] = new_activation
        new_network = type(network)(**net_dict)
        new_network.load_state_dict(network.state_dict())
        return new_network

    def parameter_mutation(self, individual):

        network = individual.actor

        mut_strength = self.cfg.mutation.mutation_sd
        num_mutation_frac = 0.1
        super_mut_strength = 10
        super_mut_prob = 0.05
        reset_prob = 0.05

        model_params = network.state_dict()

        potential_keys = []
        for i, key in enumerate(model_params):
            if not 'norm' in key:
                W = model_params[key]
                if len(W.shape) == 2:
                    potential_keys.append(key)

        how_many = np.random.randint(1, len(potential_keys) + 1, 1)[0]
        chosen_keys = np.random.choice(potential_keys, how_many, replace=False)

        for key in chosen_keys:
            W = model_params[key]
            num_weights = W.shape[0] * W.shape[1]
            num_mutations = self.rng.randint(int(np.ceil(num_mutation_frac * num_weights)))
            for _ in range(num_mutations):
                ind_dim1 = self.rng.randint(W.shape[0])
                ind_dim2 = self.rng.randint(W.shape[-1])
                random_num = self.rng.uniform(0, 1)

                if random_num < super_mut_prob:
                    W[ind_dim1, ind_dim2] += self.rng.normal(0, np.abs(super_mut_strength * W[ind_dim1, ind_dim2]))
                elif random_num < reset_prob:
                    W[ind_dim1, ind_dim2] = self.rng.normal(0, 1)
                else:
                    W[ind_dim1, ind_dim2] += self.rng.normal(0, np.abs(mut_strength * W[ind_dim1, ind_dim2]))

                if W[ind_dim1, ind_dim2] > 1000000:
                    W[ind_dim1, ind_dim2] = 1000000
                if W[ind_dim1, ind_dim2] < -1000000:
                    W[ind_dim1, ind_dim2] = -1000000

        individual.train_log["mutation"] = "parameter"

        individual.actor = network
        return individual

    def architecture_mutate(self, individual):

        offspring_actor = individual.actor.clone()
        offspring_critic_1 = individual.critic_1.clone()
        offspring_critic_2 = individual.critic_2.clone()

        rand_numb = self.rng.uniform(0, 1)
        if rand_numb < self.cfg.mutation.new_layer_prob:
            offspring_actor.add_layer()
            offspring_critic_1.add_layer()
            offspring_critic_2.add_layer()
            individual.train_log["mutation"] = "architecture_new_layer"
        else:
            node_dict = offspring_actor.add_node()
            offspring_critic_1.add_node(**node_dict)
            offspring_critic_2.add_node(**node_dict)
            individual.train_log["mutation"] = "architecture_new_node"

        individual.actor = offspring_actor
        individual.critic_1 = offspring_critic_1
        individual.critic_2 = offspring_critic_2
        return individual
