import torch
from .. import util
import numpy as np

class KLMSSampler():
  def __init__(self, n_inference_steps=torch.tensor(50), n_training_steps=torch.tensor(1000), lms_order=torch.tensor(4)):
    # Create a tensor of timesteps using torch.linspace
    timesteps = torch.linspace(n_training_steps - 1, 0, n_inference_steps)

    # Assuming util.get_alphas_cumprod can be converted to a torch version.
    # Replace this with the PyTorch equivalent of the function
    #alphas_cumprod = util.get_alphas_cumprod(n_training_steps=n_training_steps)  # Ensure this returns a PyTorch tensor

    betas = torch.linspace(0.00085 ** 0.5, 0.0120 ** 0.5, n_training_steps, dtype=torch.float32) ** 2
    # Calculate alphas
    alphas = 1.0 - betas
    # Compute cumulative product along the first dimension (dim=0)
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    
    # Compute sigmas
    sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
    log_sigmas = torch.log(sigmas)

    # Interpolating log_sigmas over timesteps
    # `torch.interp` can be used, but it is not available in PyTorch directly. We can use `torch.nn.functional.interpolate` or custom interpolation.
    # Here we'll use linear interpolation manually, assuming that the input `timesteps` matches.
    #log_sigmas_interp = torch.interp(timesteps, torch.arange(n_training_steps), log_sigmas)

    # Interpolate log_sigmas over timesteps using linear interpolation
    #log_sigmas_interp = np.interp(timesteps, range(n_training_steps), log_sigmas)
    log_sigmas_interp = self.linear_interpolate(timesteps, torch.arange(n_training_steps), log_sigmas)

    # Now calculate sigmas and append a final value of 0 to the tensor
    sigmas = torch.exp(log_sigmas_interp)
    sigmas = torch.cat([sigmas, torch.tensor([0.0])])  # Appending 0 to the tensor

    # Store the attributes
    self.sigmas = sigmas
    self.initial_scale = sigmas.max()
    self.timesteps = timesteps
    self.n_inference_steps = n_inference_steps
    self.n_training_steps = n_training_steps
    self.lms_order = lms_order
    self.step_count = 0
    self.outputs = []
        


  def get_input_scale(self, step_count=None):
      if step_count is None:
          step_count = self.step_count
      sigma = self.sigmas[step_count]
      return 1 / (sigma ** 2 + 1) ** 0.5

  def set_strength(self, strength=torch.tensor(1)):
      start_step = self.n_inference_steps - int(self.n_inference_steps * strength)
      self.timesteps = torch.linspace(self.n_training_steps - 1, 0, self.n_inference_steps)
      self.timesteps = self.timesteps[start_step:]
      self.initial_scale = self.sigmas[start_step]
      self.step_count = int(start_step)

  def step(self, latents, output):
      t = self.step_count
      self.step_count += 1

      self.outputs = [output] + self.outputs[:self.lms_order - 1]
      order = len(self.outputs)
      for i, output in enumerate(self.outputs):
          # Integrate polynomial by trapezoidal approx. method for 81 points.
          x = torch.linspace(self.sigmas[t], self.sigmas[t + 1], 81)
          y = torch.ones(81)
          for j in range(order):
              if i == j:
                  continue
              y *= x - self.sigmas[t - j]
              y /= self.sigmas[t - i] - self.sigmas[t - j]
          lms_coeff = torch.trapz(y=y, x=x)
          latents += lms_coeff * output
      return latents

  def linear_interpolate(self, x, xp, fp):
      """
      Perform linear interpolation of f at x given data points (xp, fp).
      
      Parameters:
          x (Tensor): Points at which to interpolate.
          xp (Tensor): Known data points (e.g., the training steps).
          fp (Tensor): Values at the known data points.
      
      Returns:
          Tensor: Interpolated values at x.
      """
      # Ensure tensors are of the correct shape
      xp = xp.squeeze()
      fp = fp.squeeze()

      # Ensure the input is sorted
      assert torch.all(xp[:-1] <= xp[1:]), "xp must be sorted in increasing order."

      # Get the indices of xp that are less than or equal to x
      indices = torch.searchsorted(xp, x, right=True) - 1
      
      # Handle boundary conditions (e.g., if x is out of the xp range)
      indices = torch.clamp(indices, 0, len(xp) - 2)

      # Get the left and right values for interpolation
      x_left = xp[indices]
      x_right = xp[indices + 1]
      f_left = fp[indices]
      f_right = fp[indices + 1]
      
      # Calculate the slope for linear interpolation
      slope = (f_right - f_left) / (x_right - x_left)
      
      # Perform the interpolation
      return f_left + slope * (x - x_left)
