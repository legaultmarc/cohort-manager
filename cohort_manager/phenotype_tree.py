"""
Tree representation for phenotypes.

There are different hierachical representations for phenotypes. It can be from
the questionnaire structure, from ICD10 codes or from the correlation
structure (e.g. hierarchical clustering). This module provides a generic
data structure and accessory functions to build the tree using different
methods.

TODO For now, it's only for the explicit tree.

"""


from __future__ import print_function
import collections
import subprocess

from six import StringIO as StringIO


PHENOTYPE_COLUMNS = ("name", "snomed_ct_concept", "parent", "variable_type",
                     "description", "code_name")


class NodeData(object):
    def __init__(self, *cols):
        for k, v in zip(PHENOTYPE_COLUMNS, cols):
            self.__setattr__(k, v)


class Node(object):
    def __init__(self, data=None, parent=None, children=None):
        self.data = data  # Instance of NodeData
        self.parent = parent  # Node
        self.children = [] if children is None else children  # list of Nodes

    def __repr__(self):
        return "<Node '{}' (root: {}, n_children: {})>".format(
            self.data.name if self.data else "",
            self.parent is None,
            len(self.children)
        )


class PhenotypeTree(object):
    def __init__(self, root):
        # root can be a Node instance or a list of Node instances if we have
        # multiple root phenotypes (distinct trees).
        # We always represent it as if there were many root phenotypes.
        try:
            iter(root)
        except TypeError:
            root = [root]

        self.roots = root
        for root in self.roots:
            assert isinstance(root, Node)

    def pretty_print(self, pager=False):
        """Draw the tree in the terminal window."""
        s = StringIO()
        for root in self.roots:
            self._pretty_print_subtree(root, string=s)

        s = s.getvalue().rstrip()
        if pager:
            proc = subprocess.Popen(["less"], stdin=subprocess.PIPE)
            proc.communicate(input=s.encode("utf-8"))
        else:
            print(s)

    def _pretty_print_subtree(self, node, string, depth=0):
        """Print a node and it's children."""
        bullets = ["+", "-"]
        name = "{}{} ({})".format(
            node.data.name,
            "*" if node.data.dummy else "",
            node.data.variable_type
        )

        if depth > 0:
            print(
                "\t" * depth, bullets[depth % len(bullets)], name, file=string
            )
        else:
            print(name, file=string)

        for child in node.children:
            self._pretty_print_subtree(child, string, depth=depth + 1)

    def depth_first_traversal(self):
        stack = collections.deque(self.roots)
        while stack:
            cur = stack.popleft()
            yield cur
            stack.extendleft(cur.children[::-1])

    @property
    def leaves(self):
        """Get a list of leaves from the phenotypes tree.

        Nodes that are root and have no children are excluded.

        """
        leaves = []
        for node in self.depth_first_traversal():
            if not node.children and node.parent:
                leaves.append(node)
        return leaves


def tree_from_database(con):
    """Build a phenotype tree from a cohort manager SQL database."""

    dummy_cur = con.cursor()

    def _is_dummy(name):
        dummy_cur.execute(
            "SELECT name FROM dummy_phenotypes WHERE name=?;", (name,)
        )
        return dummy_cur.fetchone() is not None

    cur = con.cursor()
    cur.execute("SELECT * FROM phenotypes")
    roots = set()
    nodes = {}
    links = []
    for tu in cur:
        node = Node()
        node.data = NodeData(*tu)
        node.data.dummy = _is_dummy(node.data.name)

        if node.data.parent:
            links.append((node.data.name, node.data.parent))
        else:
            roots.add(node)
        nodes[node.data.name] = node

    for child, parent in links:
        try:
            parent_node = nodes[parent]
        except KeyError:
            raise ValueError(
                "Invalid relationship {} -> {}. Undefined parent.".format(
                    child, parent
                )
            )
        nodes[child].parent = parent_node
        parent_node.children.append(nodes[child])

    return PhenotypeTree(roots)
