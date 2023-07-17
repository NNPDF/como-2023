import matplotlib.pyplot as plt
from feynman import Diagram

fig = plt.figure(figsize=(10.0, 10.0))
ax = fig.add_axes([0, 0, 1, 1], frameon=False)

diagram = Diagram(ax)
inB = diagram.vertex(xy=(0, 1), marker="")
inQ = diagram.vertex(xy=(0, 0), marker="")
v = diagram.vertex(xy=(0.5, 0.5))
outQ = diagram.vertex(xy=(1, 0.5), marker="")

b = diagram.line(inB, v, style="wiggly")
inQ = diagram.line(inQ, v)
outQ = diagram.line(v, outQ)

b.text(r"$b(q)$", t=0.2, y=0.07, fontsize=40)
inQ.text(r"$q(p_1)$", t=0.2, fontsize=40)
outQ.text(r"$q(p_2)$", t=0.6, fontsize=40)

diagram.plot()
# plt.show()
plt.savefig("lo-light.png")
