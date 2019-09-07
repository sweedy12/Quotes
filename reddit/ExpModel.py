from CommentNode import *

import numpy as np

class ExpModel:

    def __init__(self,beta,tree_size):
        """
        this function initiates a new exponential model.
        :param beta: the parameter of the model
        :param tree_size: the size of the tree to create
        :return:
        """
        self.beta = beta
        self.state = [0]
        self.root = CommentNode(None,None,None,0,1,None,None)
        self.nodes_list = [self.root]
        self.id = 2
        self.tree_size = tree_size


    def get_root(self):
        """
        this function returns the root of the tree
        :return:
        """
        return self.root


    def P(self,state_denom,y):
        """
        this function calculates the probability to connect to a node with y children, given the
        state of the network and the parameter beta
        :param state: The normalization factor coming from the state of the network
        :param y: the number of children in the node to connect to.
        :param beta: the model parameter
        :return:
        """
        return np.exp(y*self.beta) / state_denom

    def calculate_state_denom(self):
        """
        This function calculates the normalization factor derived by the state of the netowrk.
        :param state: the state of the network
        :param beta: the parameter of the probabilistic model.
        :return:
        """
        count = 0
        for s in self.state:
            count += np.exp(self.beta*s)
        return count

    def get_state_p(self):
        """
        this function calculates the chance to connect to each node in the state of the model.
        :return:
        """
        p = []
        state_denom = self.calculate_state_denom()
        for s in self.state:
            p.append(self.P(state_denom,s))
        return p

    def add_node(self):
        """
        this function adds a node to the tree
        :return:
        """
        p = self.get_state_p()
        #deciding which node to connect to:
        node_i = np.random.choice(len(self.state), p = p)
        father = self.nodes_list[node_i]
        #creating node:
        node = CommentNode(father.get_id(),None,None,father.get_depth()+1,self.id,None,None)
        #updating father son relations:
        father.add_child(node)
        self.id += 1
        self.state.append(0)
        self.state[node_i] += 1
        self.nodes_list.append(node)

    def build_tree(self):
        """
        This function builds the tree represented by this model
        :return:
        """
        for i in range(self.tree_size - 1): #building all the nodes
            self.add_node()


