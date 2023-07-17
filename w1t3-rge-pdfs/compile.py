import pathlib

import eko
from ekobox.cards import example

if __name__ == "__main__":
    th_card = example.theory()
    op_card = example.operator()
    # here we replace the grid with a very minimal one, to speed up the example
    op_card.xgrid = [1e-3, 1e-2, 1e-1, 5e-1, 1.0]
    op_card.n_integration_cores = 1

    path = pathlib.Path("./myeko.tar")
    path.unlink(missing_ok=True)
    eko.solve(th_card, op_card, path)