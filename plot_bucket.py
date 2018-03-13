import matplotlib.pyplot as plt
data = [191, 172, 193, 175, 184, 198, 191, 182, 181, 191, 202, 213, 189, 203, 208, 184, 203]

labels = [str(i) for i in range(-8, 9)]

plt.bar(range(len(data)), data, tick_label=labels)
plt.xlabel('delay')
plt.ylabel('number of video')
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()
