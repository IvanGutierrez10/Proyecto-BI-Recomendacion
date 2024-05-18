import numpy as np
import pandas as pd
import itertools
import time
from collections import defaultdict


class Recommender:
    """
        This is the class to make recommendations.
        The class must not require any mandatory arguments for initialization.
    """
    def __init__(self):
        self.prices = []
        self.database = []
        self.rules = []
        self.num_transacciones = []

    def eclat_recursive(self, P, minsup, F):
        for Xa, t_Xa in P.items():
            F.append((sorted(list(Xa)), len(t_Xa)))
            Pa = {}
            for Xb, t_Xb in P.items():
                if list(Xb) > list(Xa):  # Ensure correct ordering for comparison
                    Xab = Xa | Xb
                    t_Xab = t_Xa & t_Xb
                    if len(t_Xab) >= minsup:
                        Pa[Xab] = t_Xab
            if Pa:
                self.eclat_recursive(Pa, minsup, F)

    def eclat(self, db, minsup):
        # Initialize the variables
        F = []
        P = defaultdict(set)

        # Count item occurrences to filter out non-frequent items
        item_counts = defaultdict(int)
        for transaction in db:
            for item in transaction:
                item_counts[item] += 1

        # Only consider frequent items
        for i, transaction in enumerate(db):
            for item in transaction:
                if item_counts[item] >= minsup:
                    P[frozenset([item])].add(i)

        # Start the recursive Eclat algorithm
        start_time = time.time()
        self.eclat_recursive(P, minsup, F)
        end_time = time.time()

        print(f"Eclat Runtime: {end_time - start_time} seconds")
        return F
    
    def getStrongRulesFromFrequentSets(self, fsets, minconf):
        R = []
        for Z, supZ in fsets:
            if len(Z) >= 2:
                A = [set(x) for x in itertools.chain(*[itertools.combinations(Z, i) for i in range(1, len(Z))])]
                while A:
                    X = max(A, key=len)
                    A = [a for a in A if a != X]
                    X = set(X)
                    Y = set(Z) - X
                    supX = next(sup for items, sup in fsets if set(items) == X)
                    supY = next(sup for items, sup in fsets if set(items) == Y)
                    c = supZ / supX
                    if c >= minconf:
                        # Calculate lift
                        lift = c / (supY / self.num_transactions)
                        # Calculate leverage
                        leverage = (supZ / self.num_transactions) - ((supX / self.num_transactions) * (supY / self.num_transactions))
                        # Calculate Jaccard
                        jaccard = (supZ / self.num_transactions) / ((supX / self.num_transactions) + (supY / self.num_transactions) - (supZ / self.num_transactions))
                        R.append((sorted(list(X)), sorted(list(Y)), supZ, c, lift, leverage, jaccard))
                    else:
                        A = [a for a in A if not X.issubset(set(a))]
        return R
    
    def train(self, prices, database) -> None:
        """
            allows the recommender to learn which items exist, which prices they have, and which items have been purchased together in the past
            :param prices: a list of prices in USD for the items (the item ids are from 0 to the length of this list - 1)
            :param database: a list of lists of item ids that have been purchased together. Every entry corresponds to one transaction
            :return: the object should return itself here (this is actually important!)
        """
        start_time = time.time()

        self.prices = prices
        self.database = database
        self.num_transactions = len(database)

        minsup = 20  # Example value for minimum support
        minconf = 0.35  # Example value for minimum confidence

        # Find frequent itemsets
        frequent_itemsets = self.eclat(database, minsup)

        # Convert frequent itemsets to the required format for getStrongRulesFromFrequentSets
        fsets = [(set(items), sup) for items, sup in frequent_itemsets]

        # Find strong rules
        self.rules = self.getStrongRulesFromFrequentSets(fsets, minconf)

        end_time = time.time()
        print(f"Training Runtime: {end_time - start_time} seconds")

        return self

    def get_recommendations(self, cart: list, max_recommendations: int) -> list:
        """
        Makes a recommendation to a specific user.
        :param cart: a list with the items in the cart
        :param max_recommendations: maximum number of items that may be recommended
        :return: list of at most `max_recommendations` items to be recommended
        """
        start_time = time.time()
        
        recommendations = []
        cart_set = set(cart)

        # Find applicable rules
        for premise, conclusion, support, confidence, lift, leverage, jaccard in self.rules:
            if set(premise).issubset(cart_set):
                for item in conclusion:
                    if item not in cart_set and item not in [rec[0] for rec in recommendations]:
                        recommendations.append((item, confidence, lift, leverage, jaccard))
                        if len(recommendations) >= max_recommendations:
                            break

        # Sort recommendations by confidence, lift, leverage, and Jaccard in descending order
        recommendations.sort(key=lambda x: (x[1], x[2], x[3], x[4]), reverse=True)
        recommendations = [rec[0] for rec in recommendations[:max_recommendations]]

        # Sort recommendations by item prices in descending order
        recommendations.sort(key=lambda x: self.prices[int(x)], reverse=True)

        end_time = time.time()
        print(f"Recommendation Runtime: {end_time - start_time} seconds")
        return recommendations
