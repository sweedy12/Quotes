"""
This file defines a class representing a comment (Node) in the comment tree in a Reddit subreddit.
"""

import time
import datetime
class CommentNode:


    def __init__(self, father_id, body, subreddit, depth,id, date, score):
        """
        this function initiates a CommentNode substance.
        :param fatherNode: the CommentNode object, which is the father of this node.
        :param body:  the body of the comment
        :param subreddit: the subreddit (general topic) of the comment
        :param depth: the depth of the node  - how far the node is from the root comment.
        :param id: the unique id of this comment
        :param date: The date specifying the creation of the comment
        :param score: the score of the comment.
        :return:
        """
        self._father_id = father_id
        self._id = id
        self._body = body
        self._subreddit = subreddit
        self._depth = depth
        self._comment_children = []
        self.subtree_size = 0
        self._date = date
        if (date):
            self._time_stamp = self.make_time_stamp()
        else:
            self._time_stamp = self._id
        self._score = score



    def get_score(self):
        """
        :return: The score of the comment.
        """
        return self._score

    def get_time_stamp(self):
        """
        return the time stamp of this node
        :return:
        """
        return self._time_stamp

    def make_time_stamp(self):
        """
        this function makes a time stamp out of the date field of this node
        :return:
        """
        date = self._date.replace("'", "")
        stamp =time.mktime(datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S').timetuple())
        return stamp

    def get_date(self):
        """

        :return:
        """
        return self._date
    def get_rank(self):
        """

        :return: the rank of this node.
        """
        return len(self._comment_children)

    def get_father_id(self):
        """
        :return: the father node of this node
        """
        return self._father_id

    def get_subtree_size(self):
        """
        :return: the subtree size of this node
        """
        return self.subtree_size

    def set_subtree_size(self, size):
        """
        this function sets the subtree size of this node to be the number given in size
        :param size: the subtree size
        :return:
        """
        self.subtree_size = size

    def update_parent_node(self, parent):
        self._parent_node = parent

    def get_parent_node(self):
        return self._parent_node

    def get_body(self):
        """
        :return: the body of this node
        """
        return self._body

    def get_subreddit(self):
        """
        :return: the subreddit of this node
        """
        return self._subreddit

    def get_depth(self):
        """
        :return: the depth of this node
        """
        return self._depth

    def set_depth(self, depth):
        self._depth =depth

    def get_children(self):
        """
        :return: the list of children of this node
        """
        return self._comment_children

    def add_child(self, new_child):
        """
        this function adds the child given at new_Child to the list of children comments of this
        node.
        :param new_child: the child to be added
        :return:
        """
        self._comment_children.append(new_child)

    def get_id(self):
        """
        :return: the unique id of this node
        """
        return self._id

    def __lt__(self, other):
        """
        this function compares the given node and
        :param other:
        :return:
        """
        return self.get_time_stamp() < other.get_time_stamp()




