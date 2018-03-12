import matplotlib.pyplot as plt
data = [21, 19, 18, 22, 22, 137, 115, 155, 132, 148, 123, 163, 136, 138, 155, 139, 156, 147, 145, 166, 134, 157, 140, 147, 156, 157, 26, 19, 17, 26, 24]

labels = [str(i) for i in range(-15, 16)]

plt.bar(range(len(data)), data, tick_label=labels)
plt.xlabel('delay')
plt.ylabel('number of video')
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()
