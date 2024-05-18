from itertools import chain, combinations
import time
from collections import defaultdict
import itertools


class Recommender:
    """
        This is the class to make recommendations.
        The class must not require any mandatory arguments for initialization.
    """
    def __init__(self):
        self.prices = []
        self.database = []
        self.rules = []
        self.num_transacciones = 0

    def eclat_recursive(self, prefix, items, frequent_itemsets, minsup):
        while items:
          item, item_indices = items.pop(0)
          item_indices = set(item_indices)
          new_prefix = prefix.union({item})
          support = len(item_indices) / self.num_transacciones
          if support >= minsup:
              frequent_itemsets[new_prefix] = support
              suffix = []
              for other_item, other_indices in items:
                  intersection = item_indices.intersection(other_indices)
                  if len(intersection) / self.num_transacciones >= minsup:
                      suffix.append((other_item, intersection))
              self.eclat_recursive(new_prefix, suffix, frequent_itemsets, minsup)

    def eclat(self, db, minsup):
        # Create vertical data format
        item_transactions = {}
        for index, transaction in enumerate(db):
            for item in transaction:
                if item in item_transactions:
                    item_transactions[item].append(index)
                else:
                    item_transactions[item] = [index]

        # Initialize recursive search
        frequent_itemsets = {}
        sorted_items = sorted(item_transactions.items(), key=lambda x: len(x[1]), reverse=True)
        self.eclat_recursive(frozenset(), sorted_items, frequent_itemsets, minsup)
        return frequent_itemsets, item_transactions
    
    def getStrongRulesFromFrequentSets(self, fsets, minconf):
        R = []
        for Z, supZ in fsets:
            if len(Z) >= 2:
                A = [set(x) for x in itertools.chain(*[itertools.combinations(Z, i) for i in range(1, len(Z))])]
                while A:
                    A1 = A.pop(0)
                    A2 = set(Z) - A1
                    supA1 = [fs for fs in fsets if fs[0] == A1][0][1]
                    conf = supZ / supA1
                    if conf >= minconf:
                        supA2 = [fs for fs in fsets if fs[0] == A2][0][1]
                        lift = conf / (supA2 / self.num_transacciones)
                        leverage = supZ / self.num_transacciones - (supA1 / self.num_transacciones) * (supA2 / self.num_transacciones)
                        jaccard = supZ / (supA1 + supA2 - supZ)
                        R.append((A1, A2, supZ, conf, lift, leverage, jaccard))
        return R

    
    def train(self, prices, database):
        self.prices = prices
        self.database = database
        self.num_transacciones = len(database)
        print(prices)
        print(database)

        start_time = time.time()

        minsup = 0.01
        minconf = 0.05

        # Find frequent itemsets
        frequent_itemsets, item_transactions = self.eclat(database, minsup)

        # Find strong rules
        self.rules = self.getStrongRulesFromFrequentSets(item_transactions, frequent_itemsets, minconf)

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
        print ("Carro de compras:", cart)
        cart_set = set(cart)


        # Find applicable rules
        for premise, conclusion, support, confidence, lift, leverage, jaccard in self.rules:
            if set(premise).issubset(cart_set):
                for item in conclusion:
                    if item not in cart_set and item not in [rec[0] for rec in recommendations]:
                        recommendations.append((item, self.prices[int(item)], confidence, lift, leverage, jaccard))
                        if len(recommendations) >= max_recommendations:
                            break

        # Sort recommendations primarily by price and then by a combination of metrics in descending order
        recommendations.sort(key=lambda x: (x[1], x[2]*0.1 + x[3]*0.5 + x[4]*0.4 + x[5]*0.3), reverse=True)
        recommendations = [rec[0] for rec in recommendations[:max_recommendations]]

        end_time = time.time()
        print(f"Recommendation Runtime: {end_time - start_time} seconds")
        print ("Recomendaciones:", recommendations)
        return recommendations
