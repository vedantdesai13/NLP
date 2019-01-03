"""This module shows PEP8 design pattern.

It is used to give coding conventions for any python code.
"""

# alphabetically import libraries
# standard library imports first

# next import third-party libraries

# next import local applications

# a 79-char ruler (max_length allowed is 79):
# ----------------------------------------------------------------------------


a_global_variable = 1

A_CONSTANT = 'const'


# 2 empty lines between top-level functions and classes
def temp_func():
    """Write docstrings for ALL public classes, funcs and methods.

    Functions use lowercase.
    """


class MyRectangle:
    """First line of a docstring is short and next to the quotes.

    Class and exception names are CapWords.

    Closing quotes are on their own line
    """

    def __init__(self, length,
                 breadth, cost=0):
        self.length = length
        self.breadth = breadth
        self.cost = cost

    # 1 empty line between in-class def'ns
    def get_perimeter(self):
        """Method and function names are lower_case_with_underscores.

        Always use self as first arg.
        """

        return 2 * (self.length + self.breadth)  # operator spacing should improve readability.

    def get_area(self):
        return self.length * self.breadth

    def total_cost(self, percent):
        area = self.get_area()
        return area * self.cost * percent

