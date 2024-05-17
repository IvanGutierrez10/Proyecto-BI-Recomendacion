import numpy as np


class Recommender:
    """
        This is the class to make recommendations.
        The class must not require any mandatory arguments for initialization.
    """
    def eclat_recursive(self, P, minsup, F):
        for Xa, t_Xa in P.items():
            F.append((sorted(list(Xa)), len(t_Xa)))
            Pa = {}
            for Xb, t_Xb in P.items():
                if list(Xb) > list(Xa):
                    Xab = Xa | Xb
                    t_Xab = t_Xa & t_Xb
                    if len(t_Xab) >= minsup:
                        Pa[Xab] = t_Xab
            if Pa:
                self.eclat_recursive(Pa, minsup, F)

    def eclat(self, db, minsup):
        F = []
        P = {frozenset([item]): set() for transaction in db for item in transaction}
        for i, transaction in enumerate(db):
            for item in transaction:
                P[frozenset([item])].add(i)

        self.eclat_recursive(P, minsup, F)
        return F
    
    def train(self, prices, database) -> None:
        """
            allows the recommender to learn which items exist, which prices they have, and which items have been purchased together in the past
            :param prices: a list of prices in USD for the items (the item ids are from 0 to the length of this list - 1)
            :param database: a list of lists of item ids that have been purchased together. Every entry corresponds to one transaction
            :return: the object should return itself here (this is actually important!)
        """
        self.prices = prices
        self.database = database

        minsup = 3
        minconf = 0.7

        frequent_itemsets = self.eclat(self.database, minsup)

        fsets = [(set(items), sup) for items, sup in frequent_itemsets]

        self.rules = self.getStrongRulesFromFrequentSets(fsets, minconf)

        return self

    def get_recommendations(self, cart:list, max_recommendations:int) -> list:
        """
            makes a recommendation to a specific user
            
            :param cart: a list with the items in the cart
            :param max_recommendations: maximum number of items that may be recommended
            :return: list of at most `max_recommendations` items to be recommended
        """
        return [42]  # always recommends the same item (requires that there are at least 43 items)
