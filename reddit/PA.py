#-------------------------------------------------imports------------------------------------------
from CommentNode import *
import numpy as np
from scipy.stats import powerlaw as pl

class PA:

    def __init__(self, tree_sizes):
        self.tree_sizes = tree_sizes




    def create_tree(self, num_of_nodes):
        id = 0
        nodes_list = []
        deg_list = []
        #creating initial node
        root = CommentNode(None,None,None,0,0,None,None)
        id += 1.
        nodes_list.append(root)
        deg_list.append(1.)
        for i in range(num_of_nodes):
            #selecting the node to connect to:
            cur_num = np.sum(deg_list)
            cur_prob = np.divide(deg_list,cur_num)
            choice = np.random.choice(len(deg_list), p=cur_prob)
            father = nodes_list[choice]
            #creating a new node
            node = CommentNode(choice,None,None,father.get_depth()+1,id,None,None)
            id += 1
            deg_list.append(1.)
            deg_list[choice] += 1
            nodes_list.append(node)
            #updating father node
            node.update_parent_node(father)
            father.add_child(node)

        return root

    def create_all_trees(self):
        """

        :return:
        """
        trees = []
        for i in range(len(self.tree_sizes)): #iterating the number of trees
            cur_size = self.tree_sizes[i]
            trees.append(self.create_tree(cur_size))

        return trees


class Aged_PA:
    def __init__(self, tree_sizes, beta):
        self.tree_sizes = tree_sizes
        self.beta = beta



    def create_tree(self, num_of_nodes):
        id = 0
        nodes_list = []
        deg_list = []
        ages = []
        #creating initial node
        root = CommentNode(None,None,None,0,0,None,None)
        id += 1.
        nodes_list.append(root)
        deg_list.append(1.)
        ages.append(1)
        for i in range(num_of_nodes):
            #selecting the node to connect to:
            #calculating probability
            ages_rate = np.power(ages, -1*self.beta)
            p_unnormalized = np.multiply(ages_rate,deg_list)
            cur_num = np.sum(p_unnormalized)
            cur_prob = np.divide(p_unnormalized,cur_num)
            reg_prob = np.divide(deg_list, np.sum(deg_list))
            choice = np.random.choice(len(deg_list), p=cur_prob)
            father = nodes_list[choice]
            #creating a new node
            node = CommentNode(choice,None,None,father.get_depth()+1,id,None,None)
            id += 1
            #updating ages for all nodes:
            ages  = np.add(ages,1)
            ages = np.append(ages,1.)
            deg_list.append(1.)
            deg_list[choice] += 1
            nodes_list.append(node)
            #updating father node
            node.update_parent_node(father)
            father.add_child(node)



        return root

    def create_all_trees(self):
        """

        :return:
        """
        trees = []
        for i in range(len(self.tree_sizes)): #iterating the number of trees
            cur_size = self.tree_sizes[i]
            trees.append(self.create_tree(cur_size))

        return trees


