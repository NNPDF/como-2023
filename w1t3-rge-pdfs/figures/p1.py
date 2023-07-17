import sys
import matplotlib.pyplot as plt
from feynman import Diagram

fig = plt.figure(figsize=(10.0, 10.0))
ax = fig.add_axes([0, 0, 1, 1], frameon=False)

diagram = Diagram(ax)
bl = diagram.vertex(xy=(0, 0), marker="")
ct = diagram.vertex(xy=(0.5, 1), marker="")
cc = diagram.vertex(xy=(0.5, 0.5))
rc = diagram.vertex(xy=(1, 0.5), marker="")

mode = sys.argv[1]

if mode == "qbarg":
    g = diagram.line(bl, cc, style="loopy")
    t = diagram.line(ct, cc)
    rc = diagram.line(cc, rc)
    rc.text(r"$q((1-z)k)$", t=0.2, fontsize=40)
    t.text(r"$\bar q(zk)$", t=0.5, y=-0.17, fontsize=40)
    g.text(r"$g(k)$", t=0.35, y=-0.07, fontsize=40)
elif mode == "qg":
    g = diagram.line(bl, cc, style="loopy")
    t = diagram.line(cc, ct)
    rc = diagram.line(rc, cc)
    rc.text(r"$\bar q((1-z)k)$", t=0.8, fontsize=40)
    t.text(r"$q(zk)$", t=0.5, y=0.17, fontsize=40)
    g.text(r"$g(k)$", t=0.35, y=-0.07, fontsize=40)
elif mode == "gg":
    g = diagram.line(bl, cc, style="loopy")
    t = diagram.line(ct, cc, style="loopy")
    rc = diagram.line(cc, rc, style="loopy")
    rc.text(r"$g((1-z)k)$", t=0.2, fontsize=40)
    t.text(r"$g(zk)$", t=0.5, y=-0.17, fontsize=40)
    g.text(r"$g(k)$", t=0.35, y=-0.07, fontsize=40)
elif mode == "qq":
    g = diagram.line(bl, cc)
    t = diagram.line(cc, ct)
    rc = diagram.line(cc, rc, style="loopy")
    rc.text(r"$g((1-z)k)$", t=0.2, fontsize=40)
    t.text(r"$q(zk)$", t=0.5, y=0.17, fontsize=40)
    g.text(r"$q(k)$", t=0.35, y=-0.07, fontsize=40)
elif mode == "qbarqbar":
    g = diagram.line(cc, bl)
    t = diagram.line(ct, cc)
    rc = diagram.line(cc, rc, style="loopy")
    rc.text(r"$g((1-z)k)$", t=0.2, fontsize=40)
    t.text(r"$\bar q(zk)$", t=0.5, y=-0.17, fontsize=40)
    g.text(r"$\bar q(k)$", t=0.65, y=0.07, fontsize=40)
elif mode == "gq":
    g = diagram.line(bl, cc)
    t = diagram.line(cc, ct, style="loopy")
    rc = diagram.line(cc, rc)
    rc.text(r"$q((1-z)k)$", t=0.2, fontsize=40)
    t.text(r"$g(zk)$", t=0.5, y=0.27, fontsize=40)
    g.text(r"$q(k)$", t=0.35, y=-0.07, fontsize=40)
elif mode == "gqbar":
    g = diagram.line(cc, bl)
    t = diagram.line(ct, cc, style="loopy")
    rc = diagram.line(cc, rc)
    rc.text(r"$\bar q((1-z)k)$", t=0.2, fontsize=40)
    t.text(r"$g(zk)$", t=0.5, y=-0.17, fontsize=40)
    g.text(r"$\bar q(k)$", t=0.65, y=0.07, fontsize=40)
else:
    raise ValueError("Unkown mode")

diagram.plot()
# plt.show()
plt.savefig(f"p{mode}.png")
