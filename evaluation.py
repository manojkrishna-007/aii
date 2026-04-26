import matplotlib.pyplot as plt

baseline = [0.5 + i*0.01 for i in range(50)]
trained = [0.6 + i*0.02 for i in range(50)]

plt.plot(baseline, label="baseline")
plt.plot(trained, label="trained")

plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend()

plt.savefig("training_curve.png")