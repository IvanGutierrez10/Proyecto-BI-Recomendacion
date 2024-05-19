from itertools import chain, combinations
import time

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
    
    def getStrongRulesFromFrequentSets(self, item_transactions, frequent_itemsets, minconf):
        rules = []

        for itemset, itemset_support in frequent_itemsets.items():
          if len(itemset) > 1:
            for consequent in itemset:
              antecedent = itemset - frozenset([consequent])
              antecedent_support = frequent_itemsets.get(antecedent, 0)
              consequent_support = frequent_itemsets.get(frozenset([consequent]), 0)

              # Calculate Confidence
              if antecedent_support != 0:
                  confidence = itemset_support / antecedent_support
              else:
                  confidence = 0

              # Calculate Lift
              if consequent_support != 0:
                  lift = confidence / consequent_support
              else:
                  lift = 0

              # Calculate Leverage
              leverage = itemset_support - (antecedent_support * consequent_support)

              # Calculate Jaccard
              antecedent_transactions = item_transactions[list(antecedent)[0]]
              consequent_transactions = item_transactions[consequent]
              intersection_size = len(set(antecedent_transactions) & set(consequent_transactions))
              union_size = len(set(antecedent_transactions) | set(consequent_transactions))
              jaccard = intersection_size / union_size if union_size != 0 else 0

              if confidence >= minconf:
                  rules.append({
                      'antecedent': antecedent,
                      'consequent': frozenset([consequent]),
                      'confidence': confidence,
                      'lift': lift,
                      'leverage': leverage,
                      'jaccard': jaccard
                  })

        return rules
    
    def powerset(self, iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

    
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

        recommendations = {}

        for item_set in self.powerset(cart):
            item_set = frozenset(item_set)
            for rule in self.rules:
                if item_set == rule['antecedent']:
                    consequents = rule['consequent']
                    for consequent in consequents:
                        if consequent not in cart:
                            composite_score = (rule['confidence'] + rule['lift'] + rule['leverage'] + rule['jaccard']) / 4
                            if consequent not in recommendations:
                                recommendations[consequent] = composite_score
                            else:
                                recommendations[consequent] = max(recommendations[consequent], composite_score)

        recommendation_list = [(item, score, self.prices[item]) for item, score in recommendations.items()]
        sorted_recommendations = sorted(recommendation_list, key=lambda x: (-x[1], -x[2]))
        recommendations = [item for item, _, _ in sorted_recommendations[:max_recommendations]]

        end_time = time.time()
        print(f"Recommendation Runtime: {end_time - start_time} seconds")
        print (recommendations)
        return recommendations
