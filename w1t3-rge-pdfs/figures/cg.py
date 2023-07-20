import matplotlib.pyplot as plt
from feynman import Diagram

fig = plt.figure(figsize=(10.0, 10.0))
ax = fig.add_axes([0, 0, 1, 1], frameon=False)

diagram = Diagram(ax)
inB = diagram.vertex(xy=(0, 1), marker="")
inG = diagram.vertex(xy=(0, 0), marker="")
vB = diagram.vertex(xy=(0.5, 0.7))
vG = diagram.vertex(xy=(0.5, 0.3))
outQ = diagram.vertex(xy=(1, 0.7), marker="")
outQb = diagram.vertex(xy=(1, 0.3), marker="")

b = diagram.line(inB, vB, style="wiggly")
g = diagram.line(inG, vG, style="loopy")
outQ = diagram.line(vB, outQ)
t = diagram.line(vG, vB)
outQb = diagram.line(outQb, vG)

b.text(r"$\gamma(q)$", t=0.35, y=0.05, fontsize=40)
outQ.text(r"$q(p_1)$", t=0.5, fontsize=40)
outQb.text(r"$\bar q(p_2)$", t=0.5, fontsize=40)
g.text(r"$g(k_1)$", t=0.35, y=-0.07, fontsize=40)

diagram.plot()
# plt.show()
plt.savefig("cg.png")
