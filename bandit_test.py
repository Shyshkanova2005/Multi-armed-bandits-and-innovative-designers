import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

files = [
    "1.txt",
    "2.txt",
    "3.txt",
    "4.txt",
    "5.txt"
]


data = [pd.read_csv(f, header=None) for f in files]


for i, df in enumerate(data):
    print(f"\n--- Іконка {i+1} ---")
    print(df.head())


epsilon = 0.1
n_variants = 5

counts = [0] * n_variants       
rewards = [0] * n_variants      
position = [0] * n_variants     

history_probability_of_winning = []           


for t in range(300):


    if np.random.rand() < epsilon:
        chosen = np.random.randint(0, n_variants)
    else:
        mean_rewards = [rewards[i] / counts[i] if counts[i] > 0 else 0 for i in range(n_variants)]
        chosen = np.argmax(mean_rewards)


    if position[chosen] >= len(data[chosen]):
        position[chosen] = 0


    row_value = int(data[chosen].iloc[position[chosen], 0])


    counts[chosen] += 1
    rewards[chosen] += row_value
    position[chosen] += 1


    total_shows = sum(counts)
    total_wins = sum(rewards)
    history_probability_of_winning.append(total_wins / total_shows)


Probability_of_winning = [rewards[i] / counts[i] if counts[i] > 0 else 0 for i in range(n_variants)]

table = pd.DataFrame({
    "Іконка": [f"Іконка {i+1}" for i in range(n_variants)],
    "Спроби": counts,
    "Виграші": rewards,
    "Ймовірність виграшу": Probability_of_winning
})

print("\nТаблиця спроб і виграшів:")
print(table)


plt.figure(figsize=(10, 6))
x = np.arange(n_variants)

plt.bar(x - 0.2, counts, width=0.4, label="Кількість показів сторінки")
plt.bar(x + 0.2, rewards, width=0.4, label="Кількість відправлених листів")

plt.xticks(x, [f"Іконка {i+1}" for i in range(n_variants)])
plt.ylabel("Кількість (покази і відправлені листи)")
plt.title("Покази сторінок та кількості відправлених листів")
plt.legend()
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(history_probability_of_winning)
plt.xlabel("Ітерація")
plt.ylabel("Ймовірність відправлення листа при показі сторінки")
plt.title("Відношення відправлених листів до показів сторінки ")
plt.grid(True)
plt.show()


best = np.argmax(Probability_of_winning)

print(f"\nНайкраща іконка: Іконка {best+1}")
