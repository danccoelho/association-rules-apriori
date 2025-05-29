# %%
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# %%
with open('Transacoes.txt', "r") as f:
    trasaction = [line.strip().split(",") for line in f.readlines()]

trasaction

# %%
te = TransactionEncoder()
te_ary = te.fit(trasaction).transform(trasaction)
df = pd.DataFrame(te_ary, columns=te.columns_)
df

# %%
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
frequent_itemsets

# %%
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
print(rules)


