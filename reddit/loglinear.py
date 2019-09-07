import numpy as np
import sqlite3
import math


class LogLinearModel:


    def __init__(self, trees):
        """
        this is the constructor for the loglinear model class
        :param trees: a list of trees to be parsed
        :return:
        """
        self.trees = trees
        self.scenarios = []


    def create_all_scenarios(self):
        """
        this function creates all scenarios
        :return:
        """
        for tree in self.trees:
            self.parse_tree(tree)


    def parse_tree(self,tree):
        #getting the tree nodes:
        nodes = self.get_tree_nodes(tree)
        nodes = self.arrange_nodes_by_time(nodes)
        #going through all nodes in their creation order
        state = []
        nodes_index = {}
        i = 0
        for node in nodes:
            if (i == 0): #first node in the tree
                state.append(0)
                nodes_index[node.get_id()] = 0
            else:
                #creating a new scenario:
                p_id = node.get_father_id()
                if (not p_id):
                    print('shit happens')
                    break
                if (p_id not in nodes_index):
                    break
                p_ind = nodes_index[p_id]

                outcome = state[p_ind]
                scenario = Scenario(state, outcome)
                #checking if scenario already exists
                was_found = False
                for j in range(len(self.scenarios)):
                    if (self.scenarios[j] == scenario):
                        #updating the existing scenario:
                        self.scenarios[j].add_outcome(outcome)
                        was_found = True
                        break
                if (not was_found): #adding new scenario to the list
                    self.scenarios.append(scenario)

                state.append(0) #adding a new node to the state
                nodes_index[node.get_id()] = i
                #updating the state according to the father id
                state[p_ind] += 1

            i += 1






    def arrange_nodes_by_time(self, nodes):
        """
        this function returns a list of the nodes given in nodes, sorted by time
        :param nodes:
        :return:
        """
        return list(sorted(nodes, key = lambda node: node.get_time_stamp()))

    def get_tree_nodes(self, tree):
        """
        this function returns all the nodes in the tree
        :param tree:
        :return:
        """
        #running BFS on the tree
        nodes = [tree]
        all_nodes = []
        while(nodes):
            cur_node = nodes.pop()
            all_nodes.append(cur_node)
            if (cur_node.get_children()):
                nodes.extend(cur_node.get_children())
        return all_nodes

    def save_all_scenarios(self, name):
        """

        :param name:
        :return:
        """
        #creating the database or connecting to it
        conn = sqlite3.connect("scenarios\\"+name+".db")
        conn.execute('''CREATE TABLE scenarios_'''+ name+'''
        (STATE TEXT PRIMARY KEY     NOT NULL,
        OUTCOMES    TEXT NOT NULL,
        PROBABILITY TEXT NOT NULL);''')
        for scenario in self.scenarios:
            #going thorugh all scenarios
            state = str(scenario.state)
            #cleaning up the state
            state = state.replace("]","").replace("[","")
            #getting the probabilities:
            prob_dict = scenario.get_probabilities()
            outcomes = []
            probs = []
            for out in prob_dict:
                outcomes.append(out)
                probs.append(prob_dict[out])
            conn.execute("INSERT INTO scenarios_"+name+" (STATE,OUTCOMES,PROBABILITY) VALUES (?,"
                                                       "?,?)",(state,str(outcomes), str(probs)))
        conn.commit()



class Scenario:

    def __init__(self, state, outcome):
        self.state = list(sorted(state[:]))
        self.outcomes = [outcome]
        self.outcome_counter = {}
        self.outcome_counter[outcome] = 1


    def add_outcome(self, outcome):
        """
        this function adds an outcome to the given scenario
        :param outcome: an outcome to be added
        :return:
        """
        #checking if outcome already exists:
        if (outcome in self.outcomes):
            self.outcome_counter[outcome] +=1
        else:
            self.outcomes.append(outcome)
            self.outcome_counter[outcome] = 1

    def get_state(self):
        """

        :return: the state of the Scenario
        """
        return self.state

    def __eq__(self, other):
        """
        this function compares a Scenario object to another scenario object, by there states.
        :param other: A Scenario object.
        :return:
        """
        other_state = other.get_state()
        x = len(self.state)
        y = len(other_state)
        if (x != y):
            return False
        else:
            for i in range(x):
                if (self.state[i] != other_state[i]):
                    return False
            return True

    def get_probabilities(self):
        """
        this function returns the probabilities given by the outcomes of this scenario,
        as a dictionary mapping between possible outcome and its observed probability.
        :return:
        """
        prob_dict = {}
        denom = 0
        for out in self.outcome_counter:
            denom += self.outcome_counter[out]

        for out in self.outcomes:
            prob_dict[out] = float(self.outcome_counter[out]) / denom
        return prob_dict

class ModelFit:

    def __init__(self, name):
        self.name = name
        self.probs_dict = {}

    def read_probs_from_table(self):
        """

        :param name:
        :return:
        """
        conn = sqlite3.connect("scenarios\\"+self.name + ".db")
        table = conn.execute("SELECT * FROM scenarios_"+self.name)
        for row in table:
            first = row[1].replace("[","").replace("]","")
            second = row[2].replace("[","").replace("]","")
            first_a = first.split(",")
            second_a = second.split(",")
            first_i = [float(f) for f in first_a]
            second_i = [float(f) for f in second_a]
            self.probs_dict[row[0]] = (first_i, second_i)


    def l2_loss(self,a,b):
        """

        :param a:
        :param b:
        :return:
        """
        return np.power(a-b,2)

    def dl2_dbeta_l2(self,beta, observed,state,outcome):
        """
        This function calculates the derivative of the l2 loss with respect to beta, in the point
        where the parameter beta = given beta
        :param beta: the given beta
        :param observed: the observed p at a point
        :param model_p: the model_p estimate at a point
        :return:
        """
        denom = 0
        denom_extended = 0
        for s in state:
            cur = np.exp(beta*s)
            denom += cur
            denom_extended += cur*s
        p = self.P(denom,outcome,beta)
        first = 2*(p-observed)
        second = ((outcome*p*denom) - (denom_extended*p))/(denom*denom)
        return first*second

    def dll_dbeta(self,beta,state, outcome):
        """
        This function calculates the derivative of the log-likelihood, in the point wherethe
        parameter beta is beta, and the state and outcome are as given.
        The derivative function is:
        dLL/dB = (outcome - sum(exp(s*B)*s) / sum(exp(s*B))
        :param beta: The parameter value
        :param state: the state of the network
        :param outcome: the outcome received by our observations.
        :return:
        """
        acu = 0.
        denom = 0
        # denom = self.calculate_state_denom(state,beta)
        for s in state: #going over all of the networks state:
            cur = np.exp(s*beta)
            acu += cur*s
            denom += cur
        # mval = 0
        # mstate = 0
        # for s in state:
        #     cur = np.exp(beta*s) / denom
        #     if (cur > mval):
        #         mval = cur
        #         mstate = s

        #calculating the derivative value:
        return (outcome - acu/denom)

    def turn_to_state(self, key):
        """
        this function turns a key into its state representation
        :param key: the key to transform
        :return:
        """
        rep = key.split(",")
        return [int(s) for s in rep]

    def gradient_descent(self, df,init_val = 1, type = 1, Niterations = 300, eta =
        0.01):
        """
        This function performs gradient descent (or ascent) of the function who's derivative is
        given by df
        :param df: The derivative function of the function to minimize (maximize), w.r.t a given
        parameter
        :param init_val: The initial value of the parameter
        :param type: descent (-1) or ascent (1)
        :param Niterations: The number of iterations
        :param eta: the learning rate
        :return:
        """
        beta = init_val
        likelihoods = []
        loss = []
        keys = list(self.probs_dict.keys())
        states = [self.turn_to_state(state) for state in keys]
        for i in range(Niterations):
            u = []
         #gradient descent iterations
            old_beta = beta
            #going through all examples, by their state keys
            for j in range(len(keys)):

                # print(j)
                key = keys[j]
                state = states[j]
                outcomes, probs = self.probs_dict[key]
                #going through all outcomes:
                for t in range(len(outcomes)):

                    out = outcomes[t]
                    prob = probs[t]
                    #update_step
                    if (j == 199):
                        stop = 1
                    g = type*eta*df(old_beta,state,out)*prob

                    if (math.isnan(g)):
                        continue
                    else:
                        u.append(g)
                # print(beta)
            if (u):
                avg = np.mean(u)
                if (avg != math.inf and avg != -math.inf):
                    beta+= avg

            if (i%20 ==0):
                # cur_l,l = self.calculate_likelihood(beta, states)
                # likelihoods.append(cur_l)
                # loss.append(l)
                # print("likelihood:" +str(cur_l))
                print("beta:"+str(beta))
                # print("loss:"+str(l))

        print(beta)
        return beta, likelihoods,loss

    def sto_gradient_descent(self, df,init_val = 1, type = 1, Niterations = 300, eta =
        0.01, batch_size = 100):
        """
        This function performs gradient descent (or ascent) of the function who's derivative is
        given by df
        :param df: The derivative function of the function to minimize (maximize), w.r.t a given
        parameter
        :param init_val: The initial value of the parameter
        :param type: descent (-1) or ascent (1)
        :param Niterations: The number of iterations
        :param eta: the learning rate
        :return:
        """
        beta = init_val
        likelihoods = []
        loss = []
        keys = list(self.probs_dict.keys())
        num_examples = len(keys)
        states = [self.turn_to_state(state) for state in keys]
        for i in range(Niterations):
            #choosing the indices of the examples to currently work on
            ind = np.random.randint(0, high = num_examples, size = batch_size)
            u = []
         #gradient descent iterations
            old_beta = beta
            #going through all examples, by their state keys
            for j in ind:

                # print(j)
                key = keys[j]
                state = states[j]
                outcomes, probs = self.probs_dict[key]
                #going through all outcomes:
                for t in range(len(outcomes)):

                    out = outcomes[t]
                    prob = probs[t]
                    g = type*eta*df(old_beta,state,out)*prob
                    if (math.isnan(g)):
                        x=1
                    else:
                        u.append(g)

                # print(beta)
            if (u):
                avg = np.mean(u)
                if (avg != math.inf and avg != -math.inf):
                    beta += np.mean(u)

            if (i%20 ==0):
                # cur_l,l = self.calculate_likelihood(beta, states)
                # likelihoods.append(cur_l)
                # loss.append(l)
                # print("likelihood:" +str(cur_l))
                print("beta:"+str(beta))
                # print("loss:"+str(l))


        return beta, likelihoods,loss



    def P(self,state_denom,y,beta):
        """
        this function calculates the probability to connect to a node with y children, given the
        state of the network and the parameter beta
        :param state: The normalization factor coming from the state of the network
        :param y: the number of children in the node to connect to.
        :param beta: the model parameter
        :return:
        """
        return np.exp(y*beta) / state_denom

    def calculate_state_denom(self, state,beta):
        """
        This function calculates the normalization factor derived by the state of the netowrk.
        :param state: the state of the network
        :param beta: the parameter of the probabilistic model.
        :return:
        """
        count = 0
        for s in state:
            count += np.exp(beta*s)
        return count

    def calculate_loss(self,state,beta):
        """
        this function
        :param state:
        :param beta:
        :return:
        """
        key = self.turn_to_key(state)
        state_denom = self.calculate_state_denom(state,beta)
        outcomes,probs = self.probs_dict[key]
        loss = 0
        for i in range(len(outcomes)):
            model_p = self.P(state_denom,outcomes[i],beta)
            loss += self.l2_loss(model_p, probs[i])
        return loss

    def turn_to_key(self, state):
        """
        this function turns the given state into a key, meaning its string representation
        :param state: the state of the network, given as an array of numbers
        :return:
        """
        res = ""
        for i in range(len(state) - 1):
            res += str(state[i]) + ", "

        res += str(state[len(state) - 1])
        return res

    def calculate_likelihood(self,beta,states):
        """
        this function calculates the log likelihood of the data.
        :param beta: the beta parameter
        :param states: the states of the net
        :return:
        """
        likelihood = 0
        loss = 0
        keys = list(self.probs_dict.keys())
        for i in range(len(keys)):
            outcomes,probs = self.probs_dict[keys[i]]
            state = states[i]
            state_denom = self.calculate_state_denom(state, beta)
            j =0
            for out in outcomes: #going through all examples for this state
                p = self.P(state_denom, out, beta)
                likelihood += np.log(p)
                loss += self.l2_loss(p,probs[j])
                j += 1
        return likelihood,loss


