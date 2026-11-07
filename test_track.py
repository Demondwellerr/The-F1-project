import matplotlib.pyplot as plt
from track import generate_track

center, left, right, curvature, width = generate_track(seed=5)

plt.plot(*left.T, "k")
plt.plot(*right.T, "k")
plt.plot(*center.T, "r--", linewidth=0.5)
plt.axis("equal")
plt.show()