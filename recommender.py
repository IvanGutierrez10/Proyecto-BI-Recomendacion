import numpy as np


class Recommender:
    """
        This is the class to make recommendations.
        The class must not require any mandatory arguments for initialization.
    """
    
    def train(self, prices, database) -> None:
        """
            allows the recommender to learn which items exist, which prices they have, and which items have been purchased together in the past
            :param prices: a list of prices in USD for the items (the item ids are from 0 to the length of this list - 1)
            :param database: a list of lists of item ids that have been purchased together. Every entry corresponds to one transaction
            :return: the object should return itself here (this is actually important!)
        """
        
        # do something
        
        # return this object again
        return self

    def get_recommendations(self, cart:list, max_recommendations:int) -> list:
        """
            makes a recommendation to a specific user
            
            :param cart: a list with the items in the cart
            :param max_recommendations: maximum number of items that may be recommended
            :return: list of at most `max_recommendations` items to be recommended
        """
        return [42]  # always recommends the same item (requires that there are at least 43 items)
