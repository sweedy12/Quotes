"""
this file is used to mine reddit for comments, first extracting DB from it
"""
import praw
import igraph as ig
import  comment_extractor as ce
import tree_creator as tc
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly
import powerlaw
import sqlite3 as sql
import re
import PA
import pickle
import loglinear as ll
import os
import math
import ExpModel as EM
import CommentNode as CN
import tree_creator as tc


import plotly.figure_factory as ff
from mpl_toolkits.mplot3d import Axes3D

#=======================================Functions-------------------------------------------------
#----------------------------------------------------Plotting functions ---------------------------
def plot_rank_by_depth(trees, remove_ones):
    """
    this function plots the average node rank by depth function.
    :param trees: list of trees.
    :param remove_ones: boolean specifying whether to remove one node trees.
    :return:
    """
    #get average node rank by depth
    nodes_rank_depth = get_avg_rank_by_depth(trees, remove_ones)
    x = []
    y = []
    #parse x and y values
    for key in nodes_rank_depth.keys():
        x.append(key)
        y.append(nodes_rank_depth[key])

    #plot and save plot
    plt.title("Average Node rank as a function of depth")
    plt.xlabel("Depth")
    plt.ylabel("Average rank")
    plt.plot(x,y)
    plt.savefig(fname = "Relations\Rank_to_depth")
    plt.close()


#-----------------------------------Assisting functions--------------------------------------------

def calculate_trees(subreddit_names, exists = False):
    """
    this function calculates and returns all the trees generated from the subreddit names,
    as well as a mapping between each subreddit name and the trees generated from it.
    This function also returns all the table names of the tables the data is written in.
    :param subreddit_names: all of the subreddit names.
    :param exists: True iff the data already exists in tables and there's no need to ask reddit
    for it again.
    :return:  2 lists (all trees, all table_names) and a map (subreddit_name->trees).
    """
    all_trees = []
    all_table_names = []
    trees_by_sub = {}
    #creating the reddit instance
    reddit = praw.Reddit(user_agent='tree_analysis by /u/sweedy12',
                     client_id='LmEGShGtjpKJIA', client_secret='g5crqJgajPMurVeOGSj8kTtbDWI',
                     username='sweedy12', password='nirmimi12')
    #going through all subreddit names
    for subreddit_name in subreddit_names:
        if (not exists): #data does not exist - mining it.
            table_names = ce.create_subreddit_table(subreddit_name, reddit)
            file = open("table_names\\"+subreddit_name+'.txt', 'w')
            for name in table_names:
                file.write(name)
                file.write("\n")

        else: #data exists, read from tables
            table_names = []
            file = open("table_names\\"+subreddit_name+'.txt', 'r')
            names = file.readlines()
            for name in names:
                table_names.append(name[:-1])
        cur_trees = tc.create_trees("reddit_new_" + subreddit_name+".db", table_names)
        trees_by_sub[subreddit_name] = cur_trees
        all_trees += cur_trees
        all_table_names += table_names
    return all_trees, trees_by_sub,all_table_names


def calculate_distribution(subreddit_names, plots,remove_ones = True):
    """
    this function plots different distributions for different parameters.
    :param subreddit_names: the names of the subreddits the trees were drawn from.
    :param plots: a vector specifying which plots to generate.
    :param remove_ones: boolean variable specifying whether to remove single-node trees.
    """

    trees, trees_by_sub, table_names = calculate_trees(subreddit_names, exists=True)
    if (plots[TREE_PLOTS]): #plot trees
        tc.create_all_tree_plots(trees, table_names)
    if (plots[NODE_HIST_BY_SUB]): #plot node number histogram
        plot_node_distribution_by_subreddit(subreddit_names, trees_by_sub, remove_ones)
    if (plots[DEPTH_NODE_HIST]): #plot depth-to-node function
        plot_depth_to_nodes(trees)
    if (plots[TD_TOTAL_AVERAGE]): #plot average td histogram
        plot_time_difference_average_hist(trees)
    if (plots[INC_VARIANCE]): #plotting increasing variance
        plot_increasing_variance(trees)
    if (plots[ALL_TREE_NODE_HIST]): #plot all tree node distribution
        plot_all_trees_node_hist(trees, remove_ones)
    if (plots[BRANCH_FACTORS_DIST]): #plotting all branching factors
        plot_all_brnaching_factors(trees, remove_ones)
    if (plots[BRANCH_TIME_HIST]): #plot branching factor as function of avg time response
        plot_branching_time_avg(trees, remove_ones)
    if (plots[PARENT_TD_HIST]): #plotting parent td histogram
        plot_parent_td_hist(trees)


def average_multiples(x,y):
    """
    this function averages the y values of similar x values, and returns both lists.
    :param x: x values
    :param y: y values
    :return:
    """
    dict = {}
    count = {}
    #counting different appearances and summing them in different dictionaries.
    for i in range(len(x)):
        cur_x = x[i]
        cur_y = y[i]
        if (cur_x) in dict.keys():
            dict[cur_x] += cur_y
            count[cur_x] += 1.
        else:
            dict[cur_x] = float(cur_y)
            count[cur_x] = 1.
    new_x = []
    new_y = []
    #averaging the values and returning new data sets
    for key in dict.keys():
        new_x.append(key)
        new_y.append(dict[key] / count[key])
    return new_x, new_y
#--------------------------------------------------------------------------------------------------
#-------------------------------------- Word Histogram functions ----------------------------------
def create_word_hist(subreddit_names):
    """
    this function creates a histogram of word appearance in the given subreddits.
    :param subreddit_names: the subreddits to parse.
    :return: a histogram of words appearance.
    """
    hist = {}
    #going through all word files and
    for subreddit_name in subreddit_names:
        file = open("Words2\\reddit_" + subreddit_name+".db_words.txt","r",encoding="utf-8")
        hist = add_words_to_hist(hist, file) #adding words from current file to the histogram.
    return hist


def add_words_to_hist(hist,file):
    """
    this function adds count for all the words in a given file, to a given historgram
    :param hist: a dictionary, representing a histogram of word appearance.
    :param file: a file containing words.
    :return:
    """
    lines = file.readlines()
    #going through all lines
    for line in lines:
        #cleaning line
        line = line.rstrip()
        line = re.sub("[,|\\.|-|_]+"," ", line)
        line = re.sub('[^0-9a-zA-Z\\s]+', "",line)
        #splitting and adding:
        splitted = line.split(" ")
        #adding the words.
        for word in splitted:
            if (word in hist.keys()):
                hist[word] +=1
            else:
                hist[word] = 1
    return hist

def create_all_words_file(subreddit_names):
    """
    this function creates all of the words files - obe for each subreddit. each words file contains
    all of the words in a subreddit.
    :param subreddit_names: list of subreddit names
    :return:
    """
    #going through all subreddit names
    for subreddit_name in subreddit_names:
        table_names = []
        #opening a file for current table
        file = open("table_names\\"+subreddit_name+'.txt', 'r')
        names = file.readlines()
        for name in names:
            table_names.append(name[:-1])
        create_words_file("reddit_"+subreddit_name+".db",table_names)

def create_words_file(db_name, table_names):
    """
    this function creates a file containing all words in the database
    :param db_name: the name of the database to parse
    :param table_names: all table names contained in the database
    :return:
    """
    #opening the file
    file = open("Words2\\"+db_name+"_words.txt", 'w', encoding='utf-8')
    conn = sql.connect(db_name)
    #parsing all tables in the database
    for table in table_names:
        write_comments_to_file(file,conn,table)

def write_comments_to_file(file, conn, table_name):
    """
    this function writes all the oomments under the table given in table_names to the given file.
    :param file: the file to write to
    :param conn: the connecting object to the database
    :param table_name: the name of the table to parse
    :return:
    """
    table = conn.execute("SELECT * FROM "+table_name)
    for row in table:
        to_write = row[1]
        to_write.encode('utf-8')
        file.write(to_write)
        file.write("\n")

def plot_hist_from_words_dict(dict):
    """
    this function plots a histogram from a word appearance dictionary
    :param dict: dictionary mapping words to number of appearances.
    :return:
    """
    values = list(dict.values())
    x = np.arange(1,len(values)+1)
    values = list(reversed(sorted(values)))
    plt.plot(np.log(x),np.log(values))
    plt.savefig(fname = "Words2\\words_loglog_hist")
    plt.close()
    plt.plot(x,values)
    plt.savefig(fname = "Words2\\words_hist")
    plt.close()

#--------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------
#----------------------------------Data mining functions -----------------------------------------
def get_avg_rank_by_depth(trees, remove_ones = False):
    """
    this function creates and returns a map, whos keys are depth_level of nodes in the trees,
    and values are the average rank of nodes at this depth.
    :param trees: the trees to calculate on
    :param remove_ones: boolean specifying whether to include single-node trees.
    :return:
    """

    depth_counter = {}
    node_rank_counter = {}
    #going through all trees
    for root in trees:
        nodes_num = root.get_subtree_size()+1
        if ((remove_ones and nodes_num > 1) or (not remove_ones)):
            #performing a bfs traversal on the tree:
            nodes = [root]
            while (nodes):
                cur_node = nodes.pop()
                depth = cur_node.get_depth()
                #updating both dictionaries
                if (depth in depth_counter.keys()): #depth was already explored
                    depth_counter[depth] +=1.
                    node_rank_counter[depth] += cur_node.get_rank()
                else: #depth is first explored.
                    depth_counter[depth] = 1.
                    node_rank_counter[depth] = float(cur_node.get_rank())
                children = cur_node.get_children()
                if (children):
                    nodes.extend(children)
    #creating the map:
    for key in depth_counter.keys():
        node_rank_counter[key] /= depth_counter[key]
    return  node_rank_counter

def get_number_of_nodes(trees, remove_ones):
    """
    this function returns a list contaning the number of nodes in each tree
    :param trees: list of trees
    :param remove_ones: boolean specifying whether to include single-node trees.
    :return:
    """
    number_of_nodes = []
    for root in trees:
        nodes_num = root.get_subtree_size() + 1
        if ((remove_ones and nodes_num > 1) or (not remove_ones)):
            number_of_nodes.append(nodes_num)
    return number_of_nodes

def get_all_tree_depth_average(trees, remove_ones = False):
    """
    this function returns a list containing the average depth of each tree
    :param trees: list of trees
    :param remove_ones: boolean specifying whether to include single-node trees.
    :return:
    """
    data = []
    #going through all trees
    for root in trees:
        nodes_num = root.get_subtree_size() + 1
        if ((remove_ones and nodes_num > 1) or (not remove_ones)):
            data.append(get_tree_depth_average(root))
    return data

def get_tree_depth_average(root):
    """
    this function returns the average depth of the tree spanned by root
    :param root: the root of the tree
    :return:
    """
    depth_sum = 0
    leaf_sum = 0
    node_queue = [root]
    #BFS iterations
    while (node_queue):
        node = node_queue.pop()
        children = node.get_children()
        if (not children): #reached a leaf
            depth_sum += node.get_depth()
            leaf_sum +=1
        else: #continute BFS
            node_queue.extend(children)
    #return average
    return float(depth_sum) / float(leaf_sum)

def get_tree_depth(root):
    """
    this function returns the average depth of the tree spanned by root
    :param root: the root of the tree
    :return:
    """
    depth = 0
    leaf_sum = 0
    node_queue = [root]
    #BFS iterations
    while (node_queue):
        node = node_queue.pop()
        children = node.get_children()
        if (not children): #reached a leaf
            depth= max(depth,node.get_depth())
            leaf_sum +=1
        else: #continute BFS
            node_queue.extend(children)
    #return average
    return depth

def get_tree_nodes(root):
    """
    returns the list of nodes in the tree spanned by root
    :param root: the root of the tree
    :return:
    """
    nodes = [root]
    all_nodes = []
    #BFS iterations
    while (nodes):
        cur_node = nodes.pop()
        all_nodes.append(cur_node)
        children = cur_node.get_children()
        if (children):
            nodes.extend(children)
    return all_nodes

def get_branch_factors(nodes_list):
    """
    this function returns the list of ranks of the nodes in the given list
    :param nodes_list: list of all the nodes in the tree
    :return:
    """
    factors = []
    #going through all nodes and appending rank
    for node in nodes_list:
        factors.append(node.get_rank())

    return factors

def get_all_branching_factors(trees, remove_ones = False):
    """
    this function returns a list containing the average rank for each tree in trees
    :param trees: list of trees
    :param remove_ones: boolean, specifying whether to discard one node trees.
    :return:
    """
    factors = []
    for tree in trees:
        nodes_num = tree.get_subtree_size()+1
        if ((remove_ones and nodes_num > 1) or (not remove_ones)):
            nodes = get_tree_nodes(tree)
            factors.append(np.mean(get_branch_factors(nodes)))
    return factors

#--------------------------------------Node variable connections-----------------------------------
def plot_node_rank_to_depth(nodes, name):
    """
    this function plots the rank to depth function of the nodes
    :param nodes: list of nodes
    :param name: the name of the plot
    :return:
    """
    ranks = mine_rank(nodes)
    depth = mine_depth(nodes)
    ranks, depth = average_multiples(ranks,depth)
    p = np.argsort(ranks)
    ranks = [ranks[i] for i in p]
    depth = [depth[i] for i in p]
    plt.title("depth to rank function for " + name)
    plt.xlabel("rank")
    plt.ylabel("depth")
    plt.plot(ranks,depth)
    plt.savefig(fname = "Var_connections//" + name)
    plt.close()

def plot_rank_to_depth_by_time(trees):
    """
    this function plots all rank to depth functions for the nodes divided by time
    :param trees: list of trees
    :return:
    """
    hour_dict = divide_nodes_by_hour(trees)
    name = "depth_to_rank"
    for key in hour_dict.keys():
        #plotting rank to depth
        plot_node_rank_to_depth(hour_dict[key], name+"_"+key)


#--------------------------------------------------------------------------------------------------
#---------------------------------------------------Plotting functions---------------------------
def plot_histogram(data, name, is_log = False):
    """
    this function plots a histogram of the data in a file named name
    :param data: the data to plot the histogram of
    :param name: the name of the file to save
    :param is_log
    :return:
    """
    if (is_log):
        hist = [go.Histogram(x=np.log(data),histnorm="probability")]
    else:
        hist = [go.Histogram(x=data, histnorm="probability")]
    if (is_log):
        name+= "_log_scales"
        layout = go.Layout(xaxis=dict( type='log',autorange=True),
                yaxis=dict(type='log',autorange=True))
        fig = go.Figure(data = hist, layout = layout)
        plotly.offline.plot(fig, filename=name+".html", auto_open=False)
    else:
        plotly.offline.plot(hist, filename="new_Tv//"+name+".html", auto_open=False)

def plot_log_x_hist(data,name):
    hist = [go.Histogram(x=data,histnorm="probability")]
    layout = go.Layout(xaxis=dict( type='log',autorange=True),
                yaxis=dict(autorange=True))
    fig = go.Figure(data = hist, layout = layout)
    plotly.offline.plot(fig, filename=name+".html", auto_open=False)

    plotly.offline.plot(hist, filename=name+".html", auto_open=False)



def plot_var_dependency(v1,v2,trees, remove_ones = False):
    """
    this function plots v2 as a function of v1
    :param v1: first variable, one of known constants.
    :param v2: second variable, one of known constants.
    :param trees: list of trees
    :param remove_ones: boolean, specifying whether to discard one node trees.
    :return:
    """
    #getting the variables data:
    x = DATA_MINE_VEC[v1](trees, remove_ones)
    y = DATA_MINE_VEC[v2](trees, remove_ones)
    x,y = average_multiples(x,y)
    #sorting x and y accordingly:
    p = np.argsort(x)
    new_x = [x[i] for i in p]
    new_y = [y[i] for i in p]
    x_t = set(x)
    print(len(x))
    print(len(x_t))
    #getting the variable names:
    x_name = VAR_NAMES[v1]
    y_name = VAR_NAMES[v2]
    #plotting the data:
    plt.title(x_name + " as a function of " + y_name)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.scatter(new_x,new_y)
    plt.plot(new_x,new_y)
    plt.savefig("Relations\\"+x_name+"_"+y_name+"_wo_ones")
    plt.close()

def plot_parent_td_hist(trees):
    """
    this function plots the histogram of the average time difference of each node from its parent
    :param trees: list of all trees
    :return:
    """
    td_average = get_all_adjacent_td(trees)
    #plotting histogram
    name = "Distributions/parent_td"
    plot_histogram(td_average, name)

def plot_node_distribution_by_subreddit(subreddit_names, trees_by_sub, remove_ones = False):
    """
     this function creates a PLOTLY histogram plot to each of the subreddits given.
    :param subreddit_names: the names of the subreddits
    :param trees_by_sub: a mapping between the subreddit name and the list of trees under it.
    :return:
    """
    plotly.offline.init_notebook_mode(connected=True)
    #going through all subreddits
    for name in subreddit_names:
        cur_trees = trees_by_sub[name]
            #creating the histogram
        plot_name = "Histograms/" + name
        number_of_nodes = get_number_of_nodes(cur_trees, remove_ones)
        if (remove_ones):
            name+="_wo_ones"
        plot_histogram(number_of_nodes, plot_name)

def plot_depth_to_nodes(all_trees):
    """
    this function plots the histogram of the ratio between the average depth and the number of
    nodes in a tree.
    :param all_trees: list of trees
    :return:
    """
    depth_ratio = []
    for tree in all_trees:
        depth_ratio.append(get_tree_depth_average(tree) / float(tree.get_subtree_size()+1))

    name = "Distributions/depth_ratio"
    plot_histogram(depth_ratio, name)

def plot_all_trees_node_hist(trees, remove_ones = False, is_log = False):
    """
    this functions plots the histogram of the number of nodes in a tree
    :param trees: list of trees
    :param remove_ones: boolean, specifying whether to discard one node trees.
    :return:
    """
    nodes = []
    #going throug all trees
    for root in trees:
        nodes_num = root.get_subtree_size() + 1
        #appending number of nodes, id needed
        if ((remove_ones and nodes_num > 1) or (not remove_ones)):
            nodes.append(nodes_num)
    name = "Distributions/all_tree_node"
    #fitting power law
    if (remove_ones):
        name+="_wo_ones"
    plot_histogram(nodes, name, is_log=is_log)
    fit_power_law(nodes, "number of nodes")

def plot_time_difference_average_hist(trees, is_log = False):
    """
    this function plots the average time difference histogram
    :param trees: list of all trees
    :return:
    """
    #iterating over all trees:
    td_average = get_all_mean_td(trees)
    name = "Distributions/td_average"
    plot_histogram(td_average, name, is_log = is_log)

def plot_all_brnaching_factors(trees, remove_ones = False, is_log = False):
    """
    this function plots the histogram of the average rank of the the trees
    :param trees: list of trees
    :param remove_ones: a boolean, specifying whether to discard one node trees
    :return:
    """
    factors = get_all_branching_factors(trees, remove_ones)
    name = "Distributions/branching_factors"
    if (remove_ones):
        name+="_wo_ones"
    plot_histogram(factors, name, is_log= is_log)


def plot_increasing_variance(trees):
    """
    this function plots the function describing the variance change as we calculate the variance
    on more and more trees
    :param trees: list of trees
    :return:
    """
    x  = []
    variance = []
    node_num = []
    i = 0
    #iterating over all trees
    for tree in trees:
        node_num.append(tree.get_subtree_size()+1)
        if (i%100 == 0): #adding the variance every 100 iterations
            print(i)
            variance.append(np.var(node_num))
            x.append(i)
        i+=1

    #plotting the variance function
    plt.title("Variance as a function of number of trees added")
    plt.plot(x, variance)
    plt.savefig(fname = "Distributions/variance")
    plt.close()


def plot_branching_time_avg(trees, remove_ones = False):
    """
    this function plots the ratio between the average rank and time difference in the trees.
    :param trees: list of trees
    :param remove_ones: remove_ones: boolean, specifying whether to discard one node trees.
    :return:
    """
    time_factors = []
    for tree in trees:
        nodes_num = tree.get_subtree_size()
        if ((remove_ones and nodes_num > 1) or (not remove_ones)):
            nodes_list = get_tree_nodes(tree)
            brnaching_factors = get_branch_factors(nodes_list)
            avg_factor = np.mean(brnaching_factors)
            td = get_tree_time_differences(nodes_list)
            td_avg = np.mean(td)
            time_factors.append(avg_factor / td_avg)
    name = "Distributions/time_factors"
    if (remove_ones):
        name+="_wo_ones"
    plot_histogram(time_factors, name)

#-----------------------------------------------------------------------------------------------
#---------------------------------- Time functions-----------------------------------------------
def get_adjacent_time_difference(root, remove_ones = False):
    """
    this function returns a list containing the time differences between each comment and its
    parent in the tree.
    :param root: the root of the tree
    :param remove_ones: boolean, specifying whether to discard one node trees.
    :return:
    """
    td = []
    id_to_node = {}
    nodes = [root]
    #performing BFS
    while (nodes):
        cur_node = nodes.pop()
        id_to_node[cur_node.get_id()] = cur_node
        #updating TD of node from its parent:
        father_id = cur_node.get_father_id()
        if (father_id):
            father_node = id_to_node[cur_node.get_father_id()]
            cur_td = cur_node.get_time_stamp() - father_node.get_time_stamp()
            td.append(cur_td)
        children = cur_node.get_children()
        #extending the queue
        if (children):
            nodes.extend(children)

    return td


def get_all_adjacent_td(trees, remove_ones = False):
    """
    this function returns a list of average time difference from each node in the tree from its
    parent
    :param trees:a list of trees
    :param remove_ones: boolean, specifying whether to discard one node trees.
    :return:
    """
    td_average = []
    #going through all trees:
    for tree in trees:
        cur_td = get_adjacent_time_difference(tree)
        if (cur_td):
            td_average.append(np.mean(cur_td))
    return td_average

def get_tree_time_differences(nodes):
    """
    this function returns the time differences between each two adjacent comments in a submission
    :param nodes: the list of nodes
    :return:
    """
    #sorting the list of nodes:
    sorted_nodes = sorted(nodes)
    td = []
    for i in range(len(sorted_nodes) - 1):
        td.append((sorted_nodes[i+1].get_time_stamp() - sorted_nodes[i].get_time_stamp()) / 60)
    return td

def get_all_mean_td(trees, remove_ones = []):
    """
    this function returns the list of average time difference
    :param trees: list of trees
    :param remove_ones: boolean, specifying whether to discard one node trees.
    :return:
    """
    td_average = []
    #going through all trees
    for tree in trees:
        nodes_list = get_tree_nodes(tree)
        #appending the time difference, if needed
        cur_td = get_tree_time_differences(nodes_list)
        if (cur_td):
            td_average.append(np.mean(cur_td))
    return td_average

#--------------------------------------------------------------------------------------------------

#------------------------------------------- Node getters----------------------------------------

def get_avg_node_rank(nodes_list):
    """
    this function returns the average rank of the nodes in the given list
    :param nodes_list: a list of nodes
    :return:
    """
    ranks = [node.get_rank() for node in nodes_list]
    return np.mean(ranks)


def get_avg_node_depth(nodes_list):
    """
    this function returns the average depth of the nodes in the given list
    :param nodes_list: a list of nodes
    :return:
    """
    depth = [node.get_depth() for node in nodes_list]
    return np.mean(depth)

def get_avg_node_score(nodes_list):
    """
    this function returns the average depth of the nodes in the given list
    :param nodes_list: a list of nodes
    :return:
    """
    score = [node.get_score() for node in nodes_list]
    return np.mean(score)
#--------------------------------------------------------------------------------------------------

#------------------------------------- 2D histogram plotting---------------------------------------

def plot_nodes_time_hist(trees, hist_type, remove_ones):
    """
    this function plots the 2d histogram of number of nodes and time difference.
    :param trees: list of trees
    :param hist_type: type of histogram to plot (3d or contour)
    :param remove_ones: boolean, specifying whether to discard one node trees.
    :return:
    """
    #getting the node
    nodes_num = get_number_of_nodes(trees, remove_ones)
    td_average = get_all_mean_td(trees)
    #plotting the required histogram:
    name = "Distributions/time_node"
    if (remove_ones):
        name+= "_wo_ones"
    if (hist_type == CONTOUR_HIST):
        name += "_contour_hist"
        plot_contour_hist(nodes_num, td_average, name, "nodes","time-difference")
    if (hist_type == HIST_3D):
        name += "_3d_hist"
        plot_3d_hist(nodes_num,td_average,name)



def plot_2d_distributions(subreddit_names, plots,hist_type):
    """
    this function plots a 2d histogram, specified by the plots vector.
    :param subreddit_names: the names of the subreddits.
    :param plots: vector specifying which plots to produce.
    :param hist_type: the type of the histogram to plot.
    :return:
    """
    trees, trees_by_sub, table_names = calculate_trees(subreddit_names, exists=True)
    if (plots[NODES_TD_PLOT]): #plot nodes & td histogram
        plot_nodes_time_hist(trees, hist_type, remove_ones = True)


def plot_contour_hist(x,y,name,xname,yname):
    """
    this function plots a contour 2d histogram
    :param x: first variable
    :param y: second variable
    :param name: histogram name
    :param xname: first variable name
    :param yname: second variable name
    :return:
    """
    data = [go.Histogram2dContour (x=x, y=y, colorscale="blues", reversescale=True, xaxis='x',
                                   yaxis='y', histnorm='probability'),
    go.Scatter(
        x = x,
        y = y,
        xaxis = 'x',
        yaxis = 'y',
        mode = 'markers',
        marker = dict(
            color = 'rgba(0,0,0,0.3)',
            size = 3
        )
    ),
    go.Histogram(
        y = y,
        xaxis = 'x2' ,
        marker = dict(
            color = 'rgba(0,0,0,1)'
        )
    ),
    go.Histogram(
        x = x,
        yaxis = 'y2',
        marker = dict(
            color = 'rgba(0,0,0,1)'
        )
    )]
    layout = go.Layout(
    autosize = False,
    xaxis = dict(
        zeroline = False,
        domain = [0,0.85],
        showgrid = False
    ),
    yaxis = dict(
        zeroline = False,
        domain = [0,0.85],
        showgrid = False
    ),
    xaxis2 = dict(
        zeroline = False,
        domain = [0.85,1],
        showgrid = False
    ),
    yaxis2 = dict(
        zeroline = False,
        domain = [0.85,1],
        showgrid = False
    ),
    height = 600,
    width = 600,
    bargap = 0,
    hovermode = 'closest',
    showlegend = False
    )

    fig = go.Figure(data=data,layout=layout)
    plotly.offline.plot(fig, filename=name+'.html')


def plot_3d_hist(x,y,name):
    """
    this function plots a 3d histogram
    :param x: first variable
    :param y: second variable
    :param name: the name of this histogram
    :return:
    """
    fig = plt.figure()
    ax  = fig.add_subplot(111,projection = '3d')
    hist, xedges, yedges = np.histogram2d(x,y,bins=4, normed=True)
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)
    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dz = hist.flatten()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
    plt.savefig(name)
    plt.close()


#-------------------------------------------------------------------------------------------------


#------------------------------------------- fitting functions-------------------------------------
def fit_power_law(data, name):
    """
    this functions fits a power law distribution to the given data, while plotting appropriate plots
    :param data: the data to fit
    :param name: the name of the data
    :return:
    """
    #fitting a power law distribution to the data:
    result = powerlaw.Fit(data)
    plt.title("PDF plot of the data " + name +"," +'$\alpha' + " = " + str(result.alpha))
    fig = result.plot_pdf()
    result.power_law.plot_pdf(color = "r", linestyle = "-")
    # result.exponential.plot_pdf(color = "r", ax = fig)
    file = open("PowerLaw\\"+name+"_sigmas.txt", "w")
    file.write("sigma is " + str(result.sigma) + " for " + name+"\n")
    file.write("xmin is " + str(result.xmin) + "for " + name+"\n")
    plt.savefig(fname = "PowerLaw\\" + name)
    # plt.show()
    plt.close()


def fit_power_law_cdf(data, name):
    """

    :param data:
    :param name:
    :return:
    """
    """
    this functions fits a power law distribution to the given data, while plotting appropriate plots
    :param data: the data to fit
    :param name: the name of the data
    :return:
    """
    #fitting a power law distribution to the data:
    result = powerlaw.Fit(data)
    plt.title("CDF plot of the data " + name +"," +'$\alpha' + " = " + str(result.alpha))
    fig = result.plot_pdf()
    result.power_law.plot_pdf(color = "r", linestyle = "-")
    # result.exponential.plot_pdf(color = "r", ax = fig)
    file = open("PowerLaw\\"+name+"_CDF_sigmas.txt", "w")
    file.write("sigma is " + str(result.sigma) + " for " + name+"\n")
    file.write("xmin is " + str(result.xmin) + "for " + name+"\n")
    plt.savefig(fname = "PowerLaw\\_CDF_" + name)
    # plt.show()
    plt.close()


def fit_time_param(trees,param, remove_ones = False):
    time_dict = divide_by_hour(trees, remove_ones= remove_ones)
    #going through all keys
    for key in time_dict.keys():
        #fitting a power law distribution:
        cur_data = np.around(DATA_MINE_VEC[param](time_dict[key], remove_ones = remove_ones),
                                                  decimals = 1)
        name = VAR_NAMES[param]
        fit_power_law(cur_data, name+"_"+key)

def fit_node_time_param(trees, param):
    time_dict = divide_nodes_by_hour(trees)
    #going through all keys
    for key in time_dict.keys():
        #getting the data and fitting a powerlaw distribution
        cur_data = ALL_NODE_MINE_VEC[param](time_dict[key])
        name = NODE_PAR_NAMES[param]
        fit_power_law(cur_data, name+"_"+key)


#--------------------------------------------------------------------------------------------------

#----------------------------------------- Time sliders -------------------------------------------

def divide_by_hour(trees, remove_ones = False):
    """
    this function returns a mapping between an hour, and the trees created in this hour.
    :param trees: the trees to divide to hours
    :param remove_ones: boolean specifying whether to remove trees of one nodes from calculations.
    :return:
    """
    hour_dict = {}
    #going through all trees:
    for tree in trees:
        nodes_num = tree.get_subtree_size() + 1
        if ((remove_ones and nodes_num > 1) or (not remove_ones)):
            hour = parse_hour_from_date(tree.get_date())
            #checking if hour already exists:
            if (hour in hour_dict.keys()):
                hour_dict[hour].append(tree)
            else: #hour was never visited:
                hour_dict[hour] = [tree]

    return hour_dict

def divide_nodes_by_hour(trees):
    """
    this function divides all nodes in all trees by their posting hour.
    :param trees: list of trees
    :return: a mapping between posting hour and node.
    """
    hour_dict = {}
    for tree in trees:
        #getting the nodes list:
        nodes_list = get_tree_nodes(tree)
        for node in nodes_list:
            hour = parse_hour_from_date(node.get_date())
            #checking if hour exists in the data:
            if (hour in hour_dict.keys()):
                hour_dict[hour].append(node)
            else: #doesn't exist:
                hour_dict[hour] = [node]
    return hour_dict

def create_time_sliding_histogram(trees, param, remove_ones):
    """
    this function creates a histogram of the parameter given in para. This histogram is divided
    to many different histograms, divided by the hour the tree was created.
    :param trees: list of trees.
    :param param: the parameter of the histogram.
    :param remove_ones: boolean, specifying whether to discard one node trees.
    :return:
    """
    trees_by_hour = divide_by_hour(trees, remove_ones)
    mine_func = DATA_MINE_VEC[param]
    sorted_keys = sorted(trees_by_hour.keys())
    #getting the data:
    data = []
    for hour in sorted_keys:
        cur_params = mine_func(trees_by_hour[hour], remove_ones)
        data.append(go.Histogram(x= cur_params, visible=False, name=hour))
    #setting the slider:
    steps = []
    for i in range(len(data)):
        step = dict(
            method = 'restyle',
            args = ['visible', [False] * len(data)], label = sorted_keys[i]
        )
        step['args'][1][i] = True # Toggle i'th trace to "visible"
        steps.append(step)
    sliders = [dict(
        active = 10,
        currentvalue = {"prefix": "Time: ", "visible": True},
        pad = {"t": 50},
        steps = steps
    )]
    layout = dict(sliders=sliders)

    fig = dict(data=data, layout=layout)

    #setting name and plotting the figure
    name = "Sliders\\" + VAR_NAMES[param] + "_time_slider"
    if (remove_ones):
        name += "wo_ones"
    plotly.offline.plot(fig, filename=name + ".html", auto_open=False)

def plot_param_by_hour(trees, param):
    """
    this function plots a time sliding histogram of the given parameter, slided by hours.
    :param trees: list of trees.
    :param param: the parameter to plot histogram of.
    :return:
    """
    #getting the hour dict:
    hour_dict = divide_nodes_by_hour(trees)
    keys = sorted(hour_dict.keys())
    param_avg = []
    #getting the average of the wanted parameter
    for key in keys:
        param_avg.append(NODE_MINE_VEC[param](hour_dict[key]))
    #plotting:
    name = NODE_PAR_NAMES[param]
    plt.title(name + " as a function of time posted")
    plt.xlabel("time posted")
    plt.ylabel(name)
    keys_num = [float(key) for key in keys]
    plt.plot(keys_num, param_avg)
    plt.savefig(fname = "Param_to_time\\"+name)
    plt.close()

def parse_hour_from_date(date):
    """
    this function gets a date and returns the hour it represents.
    :param date: a string representing a date and hour of comment creation
    :return:
    """
    time = date.split()[1]
    hour = time[:2]
    return hour
#--------------------------------------------------------------------------------------------------


#=========================================Constants===============================================

#------------------------------------------------ 2D Histogram constants--------------------------
CONTOUR_HIST = "contour"
HIST_3D = "3d_hist"
#--------------------------------------------------------------------------------------------------
#------------------------------------------Node Mining 2-------------------------------------------

def mine_depth(nodes_list):
    """
    this function gets a list of nodes and returns a list of their depths.
    :param nodes_list: a list of nodes
    :return:
    """
    return [node.get_depth() for node in nodes_list]

def mine_rank(nodes_list):
    """
    this function gets a list of nodes and returns a list of their ranks.
    :param nodes_list: a list of nodes.
    :return:
    """
    return [node.get_rank() for node in nodes_list]

def mine_score(nodes_list):
    """
    this function gets a list of nodes and returns a list of their ranks.
    :param nodes_list: a list of nodes.
    :return:
    """
    return [node.get_score() for node in nodes_list]

#------------------------------------------------------------------------------------------------
#------------------------------------------ Score Functions ------------------------------------

def get_average_score(root, state = 0):
    """
    this function returns the average score of all the nodes in the root spanned by root.
    :param root: the root of the tree
    :return:
    """
    nodes = [root]
    num_of_nodes = 0.
    score_count  = 0.
    #BFS
    while (nodes):
        cur_node  = nodes.pop()
        first_cond = state ==0
        second_cond = (state ==1 and cur_node.get_score() > 0)
        third_cond = (state ==2 and cur_node.get_score() < 0)
        if (first_cond or second_cond or third_cond):
            num_of_nodes += 1
            score_count += cur_node.get_score()
        children = cur_node.get_children()
        if (children): #updating the nodes queue
            nodes.extend(children)
    #returning average:
    if (score_count == 0 or num_of_nodes == 0):
        return 0
    return score_count / num_of_nodes


def get_all_average_score(trees, remove_ones = False):
    """
    this function returns a list containing all the average scores of the trees given.
    :param trees: a list of trees
    :param remove_ones: a boolean, specifying whether to remove one node trees.
    :return:
    """
    average_score = [get_average_score(root) for root in trees]
    return average_score

def get_all_positive_score(trees, remove_ones = False):
    """
    this function returns a list containing all the average scores of the trees given,
    where negative score are not included.
    :param trees:  a list of trees
    :param remove_ones: a boolean, specifying whether to remove one node trees.
    :return:
    """
    average_score = [get_average_score(root, 1) for root in trees  ]
    return average_score

def get_all_negative_score(trees, remove_ones = False):
    """
    this function returns a list containing all the average scores of the trees given,
    where positive score are not included.
    :param trees:  a list of trees
    :param remove_ones: a boolean, specifying whether to remove one node trees.
    :return:
    """
    average_score = [get_average_score(root, 2) for root in trees]
    return average_score


#------------------------------------------------------------------------------------------------
#------------------------------------------- New TV data analysis -------------------------------
def get_all_scores(tree):
    """

    :param tree:
    :return:
    """
    scores = []
    nodes = [tree]
    while nodes:
        cur_node = nodes.pop()
        children = cur_node.get_children()
        if (children):
            nodes.extend(children)
        scores.append(cur_node.get_score())
    return scores

def plot_all_score_hist(trees, plotly = False):
    """

    :param trees:
    :param x_log:
    :param y_log:
    :return:
    """
    data = []
    for tree in trees:
        scores = get_all_scores(tree)
        data +=scores
    #getting the histogram of the scores
    if (plotly):
        hist = [go.Histogram(x=data)]
        plotly.offline.plot(hist, filename="New_tv//score_reg_hist_log_x_log_y", auto_open=False)
    else:
        plot_hist_all_scales(data,"all_scores")


def plot_parent_td(trees, plotly_f = False):
    """

    :param trees:
    :param plotly:
    :return:
    """
    data = []
    for tree in trees:
        data += get_adjacent_time_difference(tree)
    if (plotly_f):
        hist = [go.Histogram(x=data)]
        plotly.offline.plot(hist, filename="New_tv//all_parent_td", auto_open=False)
    else:
        plot_hist_all_scales(data,"parent_td")



def get_all_adjacent_td(trees):
    """

    :param trees:
    :return:
    """
    data = []
    for tree in trees:
        data += get_adjacent_time_difference(tree)
    return data


def plot_ranks_hists(trees):
    """

    :param trees:
    :return:
    """
    new_ranks = get_all_ranks(trees)
    plot_hist_all_scales(new_ranks,"ranks", False)
    plot_hist_all_scales_no_logbins(new_ranks, "ranks")
    #trying to fit a power law:
    compare_dists(new_ranks,"ranks")


def plot_all_hists(data, name):
    """

    :param data:
    :param name:
    :return:
    """
    plot_hist_all_scales(data, name, False)
    plot_hist_all_scales_no_logbins(data, name)
    #trying to fit a power law:
    compare_dists(data, name)

def get_all_ranks(trees):
    """

    :param trees:
    :return:
    """
    ranks = []
    for tree in trees:
        nodes = get_tree_nodes(tree)
        ranks += get_branch_factors(nodes)
    new_ranks = [r+1 for r in ranks]
    return new_ranks
    

def plot_hist_all_scales_no_logbins(data,name):

    hist,bins,_ = plt.hist(data, bins = 100)
    plt.savefig(fname = "New_tv\\" + name)
    plt.close()
    plt.hist(data, bins=100)
    plt.xscale('log')
    plt.savefig("New_tv\\" + name+"_log_x_nologbins")
    plt.close()
    plt.hist(data, bins=100)
    plt.yscale('log')
    plt.savefig("New_tv\\" + name+"_log_y_no_logbins")
    plt.close()
    plt.hist(data, bins=100)
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig("New_tv\\" + name+"_log_x_log_y_no_logbins")
    plt.close()


def plot_hist_all_scales(data, name, show = True):
    """

    :param data:
    :param name:
    :return:
    """

    #plotting the plotly histogram
    plot_histogram(data, name)
    hist,bins,_ = plt.hist(data, bins = 100)
    fnz = 0
    for i in range(len(bins)):
        if (bins[i] >= 0):
            fnz = i
            break
    if (show):
        plt.show()
    else:
        plt.savefig(fname = "EXP_trees\\" + name)
    plt.close()
    logbins = np.logspace(np.log10(bins[fnz]),np.log10(bins[-1]),len(bins))
    plt.hist(data, bins=logbins)
    plt.xscale('log')
    if (show):
        plt.show()
    else:
        plt.savefig("EXP_trees\\" + name+"_log_x")
    plt.close()
    plt.hist(data, bins=100)
    plt.yscale('log')
    if (show):
        plt.show()
    else:
        plt.savefig("EXP_trees\\" + name+"_log_y")
    plt.close()
    plt.hist(data, bins=logbins)
    plt.xscale('log')
    plt.yscale('log')
    if (show):
        plt.show()
    else:
        plt.savefig("EXP_trees\\" + name+"_log_x_log_y")
    plt.close()
    #plotting pdf:
    result = powerlaw.Fit(data)
    result.plot_pdf()
    if (show):
        plt.show()
    else:
        plt.savefig(fname = "EXP_trees\\" + name + "_pdf")
    plt.close()
    result.plot_cdf()
    if (show):
        plt.show()
    else:
        plt.savefig(fname = "EXP_trees\\" + name + "_cdf")
    plt.close()
    result.plot_ccdf()
    if (show):
        plt.show()
    else:
        plt.savefig(fname = "EXP_trees\\" + name + "_ccdf")
    plt.close()


def compare_dists(data,name):
    """

    :param data:
    :param name:
    :return:
    """
    file = open("New_tv\\"+name+"_sigmas.txt", "w")
    result = powerlaw.Fit(data)
    #plotting the different fits:
    val,p = result.distribution_compare('power_law', 'lognormal')
    file.write("power law and lognormal comparison: " + str(val) +" "+ str(p))
    file.write("\n")
    val,p = result.distribution_compare('power_law', 'exponential')
    file.write("power law and exponential comparison: " + str(val) +" "+ str(p))
    file.write("\n")
    file.write('sigma is: ' + str(result.sigma))
    file.write("\n")
    file.write("alpha is: " + str(result.alpha))
    file.write("\n")
    file.write("xmin is: " + str(result.xmin))
    #plotting the different fits:
    fig = result.plot_ccdf(linewidth = 3, label = "data")
    result.power_law.plot_ccdf(ax=fig, color = 'r', linestyle = "-", label = "powerlaw")
    result.lognormal.plot_ccdf(ax=fig, color = 'g', linestyle = "-",label = "lognormal")
    result.exponential.plot_ccdf(ax=fig, color = 'purple', linestyle = "-", label = "exponential")
    plt.legend()
    plt.savefig(fname = "New_tv\\"+ name +"_comparison")


def get_all_tree_sizes(trees):
    """

    :param trees:
    :return:
    """
    sizes = [tree.get_subtree_size() +1 for tree in trees]
    return sizes







#-------------------------------------------------------------------------------------------------
#----------------------------------------------- Constants for tree attributes ------------------
#defining a vector to hold different data extracting methods, and a vector for their names:
NODES_NUM_I = 0
LEAF_DEPTH_AVERAGE_I = 1
TOTAL_TD_AVG_I = 2
PARENT_TD_AVG_I = 3
RANK_AVG_I =4
AVG_SCORE_I = 5
POSITIVE_AVG_SCORE_I = 6
NEGATIVE_AVERAGE_SCORE_I = 7

DATA_MINE_VEC = [get_number_of_nodes,get_all_tree_depth_average,get_all_mean_td,get_all_adjacent_td,
                 get_all_branching_factors, get_all_average_score, get_all_positive_score,
                 get_all_negative_score]


VAR_NAMES = ["Number of nodes", "Depth average", "Time difference average", "Time from father "
            "comment average", "Rank average", "Score Average", "Positive score average",
             "negative_Score_average"]
#-----------------------------------------------------------------------------------------------
#--------------------------------------------------Constants for Node attributes-----------------
NODE_RANK_I = 0
NODE_DEPTH_I = 1
NODE_SCORE_I = 2

NODE_MINE_VEC = [get_avg_node_rank, get_avg_node_depth, get_avg_node_score]
ALL_NODE_MINE_VEC = [mine_rank, mine_depth, mine_score]
NODE_PAR_NAMES = ["Avg_Node_Rank", "Avg_Node_Depth", "Avg_Node_Score"]

#-------------------------Constants for different plot options -----------------------------------
TREE_PLOTS = 0
NODE_HIST_BY_SUB = 1
DEPTH_NODE_HIST = 2
TD_TOTAL_AVERAGE = 3
INC_VARIANCE = 4
ALL_TREE_NODE_HIST = 5
BRANCH_FACTORS_DIST = 6
BRANCH_TIME_HIST = 7
PARENT_TD_HIST = 8


NODES_TD_PLOT = 0
NODES_BRANCH_PLOT = 1
BRANCH_TD_PLOT = 2

#-------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
#==================================================================================================




#============================================Commands=============================================
#----------------------------------------- Defining Subreddit names-------------------------------
subreddit_names = []
subreddit_names = ['TrueDetective']
subreddit_names += ['SiliconValleyHBO', 'blackmirror', 'BoJackHorseman', 'dbz',
                   'doctorwho', 'GameOfThrones', 'MrRobot', 'Naruto', 'orangeisthenewblack',
                   'Pokemon', 'rickandmorty', 'seinfeld','southpark','startrek', 'StrangerThings',
                     'TheSimpsons', 'thewalkingdead']

subreddit_names += ['topgear','westworld','batman','starwars', 'harrypotter','lotr']
subreddit_names += [ 'marvelstudios', 'DC_Cinematic',
                   'OnePunchMan']
subreddit_names += ['DunderMifflin', 'Sherlock', 'yugioh','MakingaMurderer',]
subreddit_names += [ 'trailerparkboys','SuperNatural', 'twinpeaks','TheLastAirbender' ]
subreddit_names += ["tifu","self", "confession","fatpeoplestories", "talesfromtechsupport",
                    "talesfromretail","techsupportmacgyver","idontworkherelady",
                     "TalesFromYourServer"]
subreddit_names+= ["KitchenConfidential","TalesFromThePizzaGuy",
                     "TalesFromTheFrontDesk","pettyrevenge","prorevenge", "nosleep",
                    "LetsNotMeet","Glitch_in_the_Matrix"]
subreddit_names +=  ["shortscarystories", "thetruthishere", "UnresolvedMysteries",
                    "UnsolvedMysteries"]
subreddit_names += ["depression", "SuicideWatch", "Anxiety", "foreveralone", "offmychest",
                    "socialanxiety"]
# subreddit_names = ['seinfeld']


# #-------------------------------------------------------------------------------------------------
#-------------------------------Defining plot vector for regular distributions------------------
plots = [False]* 9
plots[TREE_PLOTS] = False
plots[NODE_HIST_BY_SUB] = True
plots[DEPTH_NODE_HIST] = True
plots[TD_TOTAL_AVERAGE] = True
plots[INC_VARIANCE] = True
plots[ALL_TREE_NODE_HIST] = True
plots[BRANCH_FACTORS_DIST] = True
plots[BRANCH_TIME_HIST] = True
plots[PARENT_TD_HIST] = True
#-------------------------------------------------------------------------------------------------


def plot_nodes_num_to_score(trees):
    """

    :param trees:
    :return:
    """
    scores = [root.get_score() for root in trees]
    nums = [root.get_subtree_size() + 1 for root in trees]
    nums, scores = average_multiples(nums,scores)
    p = np.argsort(nums)
    scores = [scores[i] for i in p]
    nums = [nums[i] for i in p]
    plt.title("number of nodes to score")
    plt.plot(nums,scores )
    plt.show()
#-----------------------------------------Commands------------------------------------------------
#---------------------------------------- model operations-----------------------------------------

def save_simple_model(name, beta, likelihoods,loss):
    """

    :param name:
    :param beta:
    :param likelihoods:
    :return:
    """
    if (math.isnan(beta)):
        beta = 11111
    #adding to the beta table:
    conn = sql.connect("simple_model\\simple_new3.db")
    conn.execute('''INSERT INTO SHOWS (SHOW,BETA) VALUES (?,?)''', (name, beta))
    conn.commit()
    #creating a directory to hold the
    os.mkdir("simple_model\\"+name)
    plt.plot(likelihoods)
    plt.xlabel("Iteration")
    plt.ylabel("Log likelihood")
    plt.savefig("simple_model\\"+name+"\\likelihood_graph_"+name)
    plt.close()
    plt.plot(np.exp(likelihoods))
    plt.xlabel("Iteration")
    plt.ylabel("exp likelihood")
    plt.savefig("simple_model\\"+name+"\\likelihood_exp_graph_"+name)
    plt.close()
    plt.plot(loss)
    plt.xlabel("Iteration")
    plt.ylabel("exp likelihood")
    plt.savefig("simple_model\\"+name+"\\loss"+name)
    plt.close()


def create_simple_table():
    """
    this function creates the table used for the simple model
    :return:
    """
    conn = sql.connect("simple_model\\simple_new3.db")
    conn.execute('''CREATE TABLE shows
        (SHOW TEXT PRIMARY KEY     NOT NULL,
        BETA    TEXT NOT NULL);''')
    conn.commit()


def create_tree_db(name):
    conn = sql.connect(name+".db")

def save_tree_to_db(db_name, tree,tree_id):
    """
    """
    name = "tree_"+str(tree_id)
    #connecting to db
    conn = sql.connect(db_name)
    #creating new table
    conn.execute('''CREATE TABLE '''+ name+'''
    (ID TEXT PRIMARY KEY     NOT NULL,
    PARENT_ID TEXT NOT NULL);''')
    print("table " + str(tree_id)+" created successfully")
    nodes_list = get_tree_nodes(tree)
    #going through all nodes
    for i in range(len(nodes_list)):
        cur_node = nodes_list[i]
        id = cur_node.get_id()
        p_id = cur_node.get_father_id()
        conn.execute("INSERT INTO "+name+" (ID,PARENT_ID) VALUES (?,?)",(str(id), str(p_id)))
    conn.commit()


def create_tree_from_table(table_name, sql_conn):
    """
    this function creates a tree from a table, representing a reddit post
    :param table_name: the name of the table to create a tree from
    :param sql_conn: an sqlite3 connector instance, connected to the database
    :return: the tree generated by the table
    """
    dict = {}
    table = sql_conn.execute("SELECT * FROM "+table_name)
    for row in table:
        if (row[0] == "1"):
            id = row[0]
            root = CN.CommentNode(None, "root", "nono", 0, id,None,None)
            dict[id] = root
        else:
            #extract parent
            #create new node:
            p_id = row[1]
            id = row[0]

            node = CN.CommentNode(p_id,None, None, 0,
                                  id,None, None)
            dict[node.get_id()] = node
    for key in dict:
        node = dict[key]
        if (node.get_father_id()):
            parent= dict[node.get_father_id()]
            node.update_parent_node(parent)
            parent.add_child(node)

    return root

#--------------------------------------------------------------------------------------------------
# #create words file and plot words histogram
# create_all_words_file(subreddit_names)
# hist = create_word_hist(subreddit_names)
# pickle.dump(hist,open("Words2\\words_hist.p","wb"))
#
# file = open("Words2\\words_hist.p", 'rb')
# hist = pickle.load(file)
# plot_hist_from_words_dict(hist)
# ce.washup("shortscarystories",460)
# trees, trees_by_sub, table_names = calculate_trees(subreddit_names, exists=True)
# plot_ranks_hists(trees)
# plot_parent_td(trees)
# plot_nodes_num_to_score(trees)
# plot_ranks_hists(trees)
# plot_all_score_hist(trees)
# data = [tree.get_subtree_size()+1 for tree in trees]
# hist, bins = np.histogram(data, bins = 100)
# plt.plot(np.log(bins[:-1]), hist)
# plt.show()

# plt.hist(data)
# plt.xscale('log')
# plt.show()
# score_Average = get_all_average_score(trees)
# positive_score_Average = get_all_positive_score(trees)
# plot_histogram(positive_score_Average, "positive score histogram")
#removing minuses:
# score_Average = [score for score in score_Average if score > 0]
# plot_histogram(score_Average, "score_average_hist")
# calculate_distribution(subreddit_names,plots, remove_ones=False)
#fitting power law to tree attribute
# data = get_all_adjacent_td(trees, False)
# fit_power_law(score_Average, "score_Average")
# #time sliding histograms
# create_time_sliding_histogram(trees,AVG_SCORE_I,False)
# #parameters by time
# plot_param_by_hour(trees, NODE_SCORE_I)
## Var dependency
# plot_var_dependency(LEAF_DEPTH_AVERAGE_I, RANK_AVG_I,trees, remove_ones=False)
# plot_rank_by_depth(trees, False)
# plots = [False]*3
# plots[NODES_TD_PLOT] = True
# calculate_distribution(subreddit_names, plots, remove_ones=False)
# plot_2d_distributions(subreddit_names,plots, hist_type=HIST_3D)
# #power law fit for time divided parameters - Trees and Nodes
# fit_time_param(trees, AVG_SCORE_I)
# fit_node_time_param(trees, NODE_SCORE_I)
# plotting rank to depth functions by time
# plot_rank_to_depth_by_time(trees)
#--------------------------------------------------------------------------------------------------
#==================================================================================================
#***********************************************************************************************
#new commands
# trees, trees_by_sub, table_names = calculate_trees(subreddit_names, exists=True)
# depths = [tree.get_depth() for tree in trees]
# print(np.average(depths))
# # data = get_all_adjacent_td(trees)
# # plot_all_hists(data,"parent_td")
# # data = get_all_tree_sizes(trees)
# x = [tree.get_subtree_size()+1 for tree in trees]
# y = [get_tree_depth(root) for root in trees]
# x,y = average_multiples(x,y)
# p = np.argsort(x)
# x = [x[i] for i in p]
# y = [y[i] for i in p]
# plt.plot(x,y,'ro')
# plt.xlabel("size")
# plt.ylabel("depth")
# plt.show()
# data = get_all_tree_sizes(trees)
# pa = PA.Aged_PA(data, 0.6)
# pa_trees = pa.create_all_trees()
# depths = get_all_tree_depth_average(pa_trees)
# plot_all_hists(depths, "depths")

# # pa_data = get_all_ranks(pa_trees)
# # plot_all_hists(pa_data, "ages_pa_100k_0_6")


# print(len(trees))
# print(len(subreddit_names))
# log_linear = ll.LogLinearModel(trees)
# log_linear.create_all_scenarios()
# prob = 2
# log_linear.save_all_scenarios("full_probs2_only_stories")

# for name in trees_by_sub:
#     cur_trees = trees_by_sub[name]
#     log_linear = ll.LogLinearModel(cur_trees)
#     log_linear.create_all_scenarios()
#     log_linear.save_all_scenarios("probs1_show__"+name)

# fitter = ll.ModelFit("full_probs2_only_stories")
# fitter.read_probs_from_table()
# beta, likelihoods,loss = fitter.gradient_descent(fitter.dll_dbeta, type=1, Niterations=3000,
#                                                 eta=0.001)
# save_simple_model("stories_all",beta,likelihoods,loss)
# plt.plot(np.multiply(np.exp(likelihoods),100))
# plt.show()

# fitter = ll.ModelFit("full_probs2_only_tv")
# fitter.read_probs_from_table()
# beta,likelihoods,loss = fitter.sto_gradient_descent(fitter.dll_dbeta, type=1,Niterations=5000,
#                                                 eta=0.0001, batch_size=5000)
# save_simple_model("tv_final", beta,likelihoods,loss)
#finding beta for all
# create_simple_table()
# for name in subreddit_names:
#      fitter = ll.ModelFit("scenarios_probs1_show__"+name)
#      fitter.read_probs_from_table()
#      beta, likelihoods,loss = fitter.sto_gradient_descent(fitter.dll_dbeta, type=1, eta=0.001,
#                                                       Niterations=5000, batch_size=1000)
#      save_simple_model(name+"_new",beta,likelihoods,loss)
# going through all trees by sub:
# conn = sql.connect("simple_model\\simple_new3.db")
# rows = conn.execute("SELECT * FROM shows;")
# beta_dict = {}
# for row in rows:
# #     beta_dict[row[0]] = row[1]
#
# sizes = [tree.get_subtree_size()+1 for tree in trees]
# # # #
# beta = 0.0912164800126477
# for i in range(80):
#     cur_trees = []
#     for size in sizes:
#         model = EM.ExpModel(beta,size)
#         model.build_tree()
#         cur_trees.append(model.get_root())
#     #saving scenarios
#     log_linear = ll.LogLinearModel(cur_trees)
#     log_linear.create_all_scenarios()
#     log_linear.save_all_scenarios("recreate"+str(i)+"_seinfeld")
#     #estimating beta again
#     fitter = ll.ModelFit("recreate"+str(i)+"_seinfeld")
#     fitter.read_probs_from_table()
#     beta,likelihoods,loss = fitter.gradient_descent(fitter.dll_dbeta, type=1,Niterations=3000,
#                                                     eta=0.001)
    # save_simple_model("recreate"+str(i)+"_seinfeld", beta,likelihoods,loss)
# for name in trees_by_sub:
#     if beta_dict[name] == str(11111):
#         continue
#     cur_trees = trees_by_sub[name]
#     lengths = [tree.get_subtree_size()+1 for tree in cur_trees]
#     #reading beta
#     beta = float(beta_dict[name])
#     #creating models
#     trees = []
#     for size in lengths:
#         model = EM.ExpModel(beta,size)
#         model.build_tree()
#         trees.append(model.get_root())
#     log_linear = ll.LogLinearModel(trees)
#     log_linear.create_all_scenarios()
#     log_linear.save_all_scenarios("probs1_show__"+name)
#
# # create_simple_table()

# db_name = "EXP_trees\\exponential_trees"
# create_tree_db(db_name)
# #beta trees recreation
# beta = 0.0177404999778128
# new_trees = []
# count = 0
# for tree in trees:
#     expmod = EM.ExpModel(beta, tree.get_subtree_size())
#     expmod.build_tree()
#     save_tree_to_db(db_name+".db",expmod.get_root(),count)
#     count += 1



# def estimate_beta_different_domains(name,c_names):
#     """
#
#     :param name:
#     :param c_names:
#     :return:
#     """
#     b1 = []
#     new_names = [c+"_new" for c in c_names]
#     conn = sql.connect("simple_model\\simple_new3.db")
#     row = conn.execute("SELECT * from shows")
#     for r in row:
#         if (r[0] in new_names):
#             if (r[1] != '11111'):
#                 b1.append(float(r[1]))
#     print(b1)
#     plt.hist(b1)
#     plt.title("Beta value distribution for category: " +name)
#     plt.savefig(fname = "beta_est\\dist_"+name)
#     variance = np.var(b1)
#     mean = np.mean(b1)
#     print(variance)
#     print(mean)
#     print(variance/mean)
#
#
#
#
# estimate_beta_different_domains("all subreddits", subreddit_names)


#PA trees:
# sizes = [tree.get_subtree_size()+1 for tree in trees]
# tree_pa = PA.PA(sizes)
# pa_trees = tree_pa.create_all_trees()
# #getting the average depth of these trees:
# average_depth = [get_tree_depth_average(root) for root in pa_trees]
# # plot_hist_all_scales(average_depth,"PA_average_depth", show=False)
# depths = [get_tree_depth(root) for root in pa_trees]
# # plot_hist_all_scales(depths,"PA_depth", show=False)
# avg_rank = [get_avg_node_rank(get_tree_nodes(root)) for root in pa_trees]
# # plot_hist_all_scales(avg_rank,"pa_avg_rank", show = False)
# all_ranks = get_all_ranks(pa_trees)
# plot_hist_all_scales(all_ranks,"pa_all_ranks",show=False)

# EXP trees
conn = sql.connect("EXP_trees\\exponential_trees.db")
exp_trees = []
for i in range(32139):
    tree = create_tree_from_table("tree_"+str(i), conn)
    tc.update_tree(tree)
    exp_trees.append(tree)



average_depth = [get_tree_depth_average(root) for root in exp_trees]
# plot_hist_all_scales(average_depth,"GRN_average_depth", show=False)
depths = [get_tree_depth(root) for root in exp_trees]
# plot_hist_all_scales(depths,"GRN_depth", show=False)
avg_rank = [get_avg_node_rank(get_tree_nodes(root)) for root in exp_trees]
# plot_hist_all_scales(avg_rank,"GRN_avg_rank", show = False)
all_ranks = get_all_ranks(exp_trees)
# plot_hist_all_scales(all_ranks,"GRN_all_ranks",show=False)


#Reddit trees
# average_depth = [get_tree_depth_average(root) for root in trees]
# # plot_hist_all_scales(average_depth,"Reddit_average_depth", show=False)
# depths = [get_tree_depth(root) for root in trees]
# # plot_hist_all_scales(depths,"Reddit_depth", show=False)
# avg_rank = [get_avg_node_rank(get_tree_nodes(root)) for root in trees]
# # plot_hist_all_scales(avg_rank,"Reddit_avg_rank", show = False)
# all_ranks = get_all_ranks(trees)

#printing vars and means
print("average_depth:"+str(np.mean(average_depth)))
print("depth:"+str(np.mean(depths)))
print("avg_rank:"+str(np.mean(avg_rank)))
print("all_ranks:"+str(np.mean(all_ranks)))
print("avg_depth_var:"+str(np.var(average_depth)))
print("depth_var:"+str(np.var(depths)))
print("avg_rank_var:"+str(np.var(avg_rank)))
print("all_ranks_var:"+str(np.var(all_ranks)))



# plot_hist_all_scales(all_ranks,"Reddit_all_ranks",show=False)

#
# def get_seinfeld_betas():
#     """
#
#     :return:
#     """
#     names = [str(i) for i in range(30,71)]
#     names += ["recreate"+str(i)+"_seinfeld" for i in range(6,30)]
#     names += ["seinfeld_final_recreate0"+str(i) for i in range(6)]
#     conn = sql.connect("simple_model\\simple_new3.db")
#     row = conn.execute("SELECT * from shows")
#     betas = []
#     for r in row:
#         if (r[0] in names):
#             if (r[1] != '11111'):
#                 betas.append(float(r[1]))
#     print(np.mean(betas))
#     print(np.var(betas))
#     print(len(betas))
#     # plot_hist_all_scales(betas, "parameter estimation", show = False)
#     plt.hist(betas)
#     plt.title("Parameter estimation histogram for subreddit 'Seinfeld'")
#     plt.savefig(fname = "EXP_trees\\betas")
#
# get_seinfeld_betas()