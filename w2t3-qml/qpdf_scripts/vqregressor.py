import os

import numpy as np
import matplotlib.pyplot as plt 

from qibo import gates, hamiltonians
from qibo.derivative import parameter_shift
from qibo.models import Circuit


class VQRegressor:

  def __init__(self, data, labels, layers, nqubits=1):
    """Class constructor of a variational quantum regressor model."""
    # some general features of the QML model
    self.nqubits = nqubits
    self.layers = layers
    self.data = data
    self.labels = labels
    self.ndata = len(labels)

    # initialize the circuit and extract the number of parameters
    self.circuit = self.ansatz(nqubits, layers)
    print(self.circuit.draw())

    # get the number of parameters
    self.nparams = len(self.circuit.get_parameters())
    # set the initial value of the variational parameters
    np.random.seed(1234)
    self.params = np.random.randn(self.nparams)
    # scaling factor for custom parameter shift rule
    self.scale_factors = np.ones(self.nparams)

    # define the observable
    self.h = hamiltonians.Z(nqubits)

# ---------------------------- ANSATZ ------------------------------------------

  def ansatz(self, nqubits, layers):
    """Here we implement the variational model ansatz."""
    c = Circuit(nqubits)
    for q in range(nqubits):
      for _ in range(layers):
        c.add(gates.RY(q=q, theta=0))
        c.add(gates.RY(q=q, theta=0))
        c.add(gates.RZ(q=q, theta=0))
        c.add(gates.RZ(q=q, theta=0))
    c.add(gates.M(0))

    return c

# --------------------------- RE-UPLOADING -------------------------------------

  def inject_data(self, x):
    """Here we combine x and params in order to perform re-uploading."""
    params = []
    index = 0

    for q in range(self.nqubits):
      for l in range(self.layers):
        # embed X
        params.append(self.params[index]*x)
        params.append(self.params[index+1])
        params.append(self.params[index+2]*x)
        params.append(self.params[index+3])
        # update scale factors
        # equal to x only when x is involved
        self.scale_factors[index] = x
        self.scale_factors[index+2] = x
        # we have three parameters per layer
        index += 4

    # update circuit's parameters
    self.circuit.set_parameters(params)


# ------------------------------- PREDICTIONS ----------------------------------

  def one_prediction(self, x):
    """This function calculates one prediction with fixed x."""
    self.inject_data(x)

    return self.h.expectation(self.circuit.execute().state())


  def predict_sample(self):
    """This function returns all predictions."""
    predictions = []
    for x in self.data:
      predictions.append(self.one_prediction(x))

    return predictions


# ------------------------ PERFORMING GRADIENT DESCENT -------------------------


  def circuit_derivative(self):
    """Derivatives of the expected value of the target observable with respect
    to the variational parameters of the circuit are performed via parameter-shift
    rule (PSR)."""
    dcirc = np.zeros(self.nparams)

    for par in range(self.nparams):
      # read qibo documentation for more information about this PSR implementation
      dcirc[par] = parameter_shift(
          circuit = self.circuit,
          hamiltonian = self.h,
          parameter_index = par,
          scale_factor = self.scale_factors[par]
          )

    return dcirc


  def evaluate_loss_gradients(self):
    """This function calculates the derivative of the loss function with respect
    to the variational parameters of the model."""

    # we need the derivative of the loss
    # nparams-long vector
    dloss = np.zeros(self.nparams)
    # we also keep track of the loss value
    loss = 0

    # cycle on all the sample
    for x, y in zip(self.data, self.labels):
      # calculate prediction
      prediction = self.one_prediction(x)
      # calculate loss
      res = (prediction - y)
      loss += res**2
      # derivative of E[O] with respect all thetas
      dcirc = self.circuit_derivative()
      # calculate dloss
      dloss += 2 * res * dcirc

    return dloss/len(self.data), loss/len(self.data)
  

  def apply_adam(
    self,
    learning_rate,
    m,
    v,
    iteration,
    beta_1=0.85,
    beta_2=0.99,
    epsilon=1e-8,
  ):
    """
    Implementation of the Adam optimizer: during a run of this function parameters are updated.
    Furthermore, new values of m and v are calculated.
    Args:
        learning_rate: np.float value of the learning rate
        m: momentum's value before the execution of the Adam descent
        v: velocity's value before the execution of the Adam descent
        features: np.matrix containig the n_sample-long vector of states
        labels: np.array of the labels related to features
        iteration: np.integer value corresponding to the current training iteration
        beta_1: np.float value of the Adam's beta_1 parameter; default 0.85
        beta_2: np.float value of the Adam's beta_2 parameter; default 0.99
        epsilon: np.float value of the Adam's epsilon parameter; default 1e-8
    Returns: np.float new values of momentum and velocity
    """

    grads, loss = self.evaluate_loss_gradients()

    m = beta_1 * m + (1 - beta_1) * grads
    v = beta_2 * v + (1 - beta_2) * grads * grads
    mhat = m / (1.0 - beta_1 ** (iteration + 1))
    vhat = v / (1.0 - beta_2 ** (iteration + 1))
    self.params -= learning_rate * mhat / (np.sqrt(vhat) + epsilon)

    return m, v, loss, grads


  def gradient_descent(
    self,
    learning_rate,
    epochs,
    J_treshold=1e-5,
    live_plotting=True,
  ):
    """
    This function performs a full gradient descent strategy.

    Args:
      learning_rate (float): learning rate.
      epochs (int): number of optimization epochs.
      batches (int): number of batches in which you want to split the training set.
      (default 1)
      restart_from_epoch (int): epoch from which you want to restart a previous
      training (default None)
      method (str): gradient descent method you want to perform. Only "Standard"
      and "Adam" are available (default "Adam").
      J_treshold (float): target value for the loss function.
    """

    # creating folder where to save params during training
    # we create a folder
    os.system("mkdir -p ./live-plotting")
    # we clean it if already exists
    os.system("rm ./live-plotting/*.png")

    # we track the loss history
    loss_history, grad_history = [], []

    # m and v values for Adam optimization
    m = np.zeros(self.nparams)
    v = np.zeros(self.nparams)

    # cycle over the epochs
    for epoch in range(epochs):

        iteration = 0

        # stop the training if the target loss is reached
        if epoch != 0 and loss_history[-1] < J_treshold:
            print(
                "Desired sensibility is reached, here we stop: ",
                iteration,
                " iteration",
            )
            break


        # update iteration tracker
        iteration += 1

        m, v, loss, grads = self.apply_adam(
            learning_rate=learning_rate, 
            m=m, 
            v=v, 
            iteration=iteration
        )

        grad_history.append(grads)
        loss_history.append(loss)

        # track the training
        print(
            "Iteration ",
            iteration,
            " epoch ",
            epoch + 1,
            " | loss: ",
            loss,
        )

        if live_plotting:
            self.show_predictions(f"Live_predictions", save=True)

    return loss_history

# ---------------------- PLOTTING FUNCTION -------------------------------------

  def show_predictions(self, title, save=False, xscale="linear"):
    """This function shows the obtained results through a scatter plot."""

    # calculate prediction
    predictions = self.predict_sample()

    # draw the results
    plt.figure(figsize=(8,5))
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(self.data, self.labels, color='purple', alpha=0.6, label='Target', s=25, marker='o')
    plt.scatter(self.data, predictions, color='orange', alpha=0.6, label='Predictions', s=25, marker='o')
    plt.xscale(xscale)
    plt.grid(True)
    plt.legend()

    # we save all the images during the training in order to see the evolution
    if save:
      plt.savefig(f'./live-plotting/'+str(title)+'.png')
      plt.close()

    plt.show()
