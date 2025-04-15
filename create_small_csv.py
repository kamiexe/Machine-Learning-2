import pandas as pd

# Read the original CSV
original_path = "../data/test.csv"
df = pd.read_csv(original_path)

# Take a 10% random sample (adjust frac=0.1 to your needs)
df_small = df.sample(frac=0.1, random_state=42)

# Save the smaller CSV
df_small.to_csv("../data/test_small.csv", index=False)

print("Small CSV created!")