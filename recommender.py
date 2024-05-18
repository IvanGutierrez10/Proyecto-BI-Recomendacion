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
        self.num_transacciones = 0

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
                    A1 = A.pop(0)
                    A2 = set(Z) - A1
                    # Find support for A1
                    supA1_list = [fs for fs in fsets if fs[0] == A1]
                    if not supA1_list:
                        continue
                    supA1 = supA1_list[0][1]
                    if supA1 == 0:
                        continue
                    conf = supZ / supA1
                    if conf >= minconf:
                        # Find support for A2
                        supA2_list = [fs for fs in fsets if fs[0] == A2]
                        if not supA2_list:
                            continue
                        supA2 = supA2_list[0][1]
                        lift = conf / (supA2 / len(self.num_transacciones))
                        leverage = supZ / len(self.num_transacciones) - (supA1 / len(self.num_transacciones)) * (supA2 / len(self.num_transacciones))
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

        minsup = 20  # Example value for minimum support
        minconf = 0.25  # Example value for minimum confidence

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
                        recommendations.append((item, self.prices[int(item)], confidence, lift, leverage, jaccard))
                        if len(recommendations) >= max_recommendations:
                            break

        # Sort recommendations primarily by price and then by a combination of metrics in descending order
        recommendations.sort(key=lambda x: (x[1], x[2]*0.4 + x[3]*0.3 + x[4]*0.2 + x[5]*0.1), reverse=True)
        recommendations = [rec[0] for rec in recommendations[:max_recommendations]]

        end_time = time.time()
        print(f"Recommendation Runtime: {end_time - start_time} seconds")
        return recommendations
