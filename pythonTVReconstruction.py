import numpy as np
import ipywidgets as widget

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
  
anglesSelector = widget.Dropdown(
    options=[('12 kulmaa', 12), ('20 kulmaa', 20), ('40 kulmaa', 40), ('60 kulmaa', 60)],
    value=20,
    description='Kulmien lukumäärä:',
    disabled=False,
)

angleTypeSelector = widget.Dropdown(
    options=[('Harvakulma', 'sparse'), ('Tiheäkulma', 'dense')],
    value='dense',
    description='Kulmien tyyppi',
    disabled=False,
)
    
alphaSelector = widget.Dropdown(
    options=[('10⁻⁸', -8), ('10⁻⁵', -5), ('10⁻²', -2), ('10', 1)],
    value=-5,
    description='Regularisaatioparametri:',
    disabled=False,
)
  
iterationsSelector = widget.Dropdown(
    options=[('10²', 100), ('10³', 1000), ('10⁴',10000), ('10⁵',100000)],
    value=100,
    description='Toistojen lukumäärä:',
    disabled=False,
)

def displayParameterWidgets():
  # Display the UI widgets
  
  global anglesSelector
  global angleTypeSelector
  global alphaSelector
  global iterationsSelector

  display(anglesSelector)
  display(angleTypeSelector)
  display(alphaSelector)
  display(iterationsSelector)

def _proximal(u, alpha, p):
  # Solves v + alpha*v^(p-1) = u for non-negative u
  if(p == 2):
    # p == 2 is the simplest case
    v = u / (1 + alpha)

  elif(p < 2):
    # p < 2 requires special initial values
    v = np.minimum(u, np.power((u / alpha * (2 - p)),(1 / (p - 1)) / 2))
    
    for i in range(10):
      v = v + ( u - v - alpha * np.power(v, (p - 1))) / (1 + alpha * (p - 1) * np.power(v, (p - 2)))

  elif(p > 2):
    # for p > 2, the best guess is to neglect the x term
    v = np.power((u / alpha), (1 / (p - 1)))

    for i in range(10):
      v = v + ( u - v - alpha * np.power(v, (p - 1))) / (1 + alpha * (p - 1) * np.power(v, (p - 2)))
      
  return v

def _dxp(u):
  # Select columns from 2 to end, add the last original column as the last and finally subtract the original matrix
  [m,n] = u.shape
  dx = np.hstack((u[:,1:], u[:,n-1:n]))
  dx = dx - u
  #dx = [u(:,2:end) u(:,end)] - u;
  return dx

def _dyp(u):
  # Select rows from 2 to end, add the last original row as the last and finally subtract the original matrix
  [m,n] = u.shape
  dy = np.vstack((u[1:,:], u[n-1:n,:]))
  dy = dy - u
  #dy = [u(2:end,:); u(end,:)] - u;
  return dy

def _dxm_ad(u):
  # Select first n-1 columns and add zeroes in the end. Then subtract a matrix with zeroes as the first column and the data from columns 1 to n-1
  [m,n] = u.shape
  dx = np.hstack((u[:,:n-1], np.zeros((m,1))))
  dx = dx - np.hstack((np.zeros((m,1)), u[:,:n-1]))
  #dx = [u(:,1:end-1) zeros(M,1)] - [zeros(M,1) u(:,1:end-1)];
  return dx
  
def _dym_ad(u):
  # Select first n-1 rows and add zeros at the bottom. The subtract a matrix with zeros as the top row and data from rows 1 to n-1
  [m,n] = u.shape
  dy = np.vstack((u[:n-1,:], np.zeros((1,n))))
  dy = dy - np.vstack((np.zeros((1,n)), u[:n-1,:]))
  #dy = [u(1:end-1,:);np.zeros(1,N)] - [np.zeros(1,N);u(1:end-1,:)];
  return dy
  
def reconstructTotalVariation(m, A, q_exp, lamb, maxits):

  # This is a Python reimplementation of the tomo_tv.m module originally by Samuli Siltanen based on the work of Kristian Breides

  # Reconstruct tomography image from the measurement matrix and sinogram
  q_exp_dual = q_exp/(q_exp - 1)

  # Set the size of the reconstructed image
  N = int(np.sqrt(A.shape[1]))
  
  u = np.zeros((N,N))
  u_ = u
  v = np.zeros(m.shape)
  px = np.zeros((N,N))
  py = np.zeros((N,N))

  # Lipschitz parameter and step length
  L2 = 8
  sigma = 1 / np.sqrt(L2)
  tau = 1 / np.sqrt(L2)

  bar = widget.IntProgress(value=0, min = 0, max = maxits, step = 1, description='Progress:', bar_style='info', orientation='horizontal')
  display(bar)

  #for i in tqdm(range(maxits)):
  for i in range(0, maxits):
    
    # Ascend step for v
    v = v + sigma * ((A * u.flatten('F')).reshape((m.shape), order = 'F') - m)
    vabs = np.abs(v)
    vabsnew = _proximal(vabs, sigma, q_exp_dual)
    I = vabs > 0
    v[I] = v[I] / vabs[I] * vabsnew[I]

    # Ascend step for p
    ux = _dxp(u_)
    uy = _dyp(u_)

    px = px + sigma * ux
    py = py + sigma * uy

    # Proximal mapping wrt p^*-norm
    pabsm = np.maximum([1], (np.power((np.power(px, 2) + np.power(py, 2)), (1/2)) / lamb))
    px = px / pabsm
    py = py / pabsm

    # Descend step
    uold = u
    adjoint = (A.transpose() * v.flatten('F')).reshape((N,N), order='F')
    div = _dxm_ad(px) + _dym_ad(py)

    u = u - tau * (adjoint - div)
    u = np.maximum([0], u)

    # Leading point
    u_ = 2 * u - uold

    bar.value = i + 1 
    
  return u

def drawPlots(images, angles, iterations):
  
  # Helper function to draw the default plots
  fig = plt.figure(1)
  fig.suptitle('Tulosten vertailu')
  spec = gridspec.GridSpec(ncols = 5, nrows = 3, figure = fig)

  fig.add_subplot(spec[0,0])
  plt.imshow(images['sinogram120'], aspect='auto')
  plt.gray()
  plt.title('Alkuperäinen sinogrammi')
  plt.autoscale()

  fig.add_subplot(spec[0,1])
  plt.imshow(images['fbp120'])
  plt.gray()
  plt.title('Takaisinprojektio, 120 kulmaa')
  plt.autoscale()

  fig.add_subplot(spec[0,4])
  plt.imshow(images['tvImgFull1'])
  plt.gray()
  plt.title('Kohinanpoisto, 120 kulmaa, 1 toisto')
  plt.autoscale()

  fig.add_subplot(spec[0,3])
  plt.imshow(images['tvImgFull10'])
  plt.gray()
  plt.title('Kohinanpoisto, 120 kulmaa, 10 toistoa')
  plt.autoscale()

  fig.add_subplot(spec[0,2])
  plt.imshow(images['tvImgFullMax'])
  plt.gray()
  plt.title('Kohinanpoisto, 120 kulmaa, ' + str(iterations) + ' toistoa')
  plt.autoscale()

  fig.add_subplot(spec[1,0])
  plt.imshow(images['sinogramSample'], aspect='auto')
  plt.gray()
  plt.title('Näytesinogrammi, ' + str(angles) + ' kulmaa')
  plt.autoscale()

  fig.add_subplot(spec[1,1])
  plt.imshow(images['fbpSample'])
  plt.gray()
  plt.title('Takaisinprojektio, ' + str(angles) + ' kulmaa')
  plt.autoscale()

  fig.add_subplot(spec[1,4])
  plt.imshow(images['tvImgSample1'])
  plt.gray()
  plt.title('Kohinanpoisto, ' + str(angles) + ' kulmaa, 1 toisto')
  plt.autoscale()

  fig.add_subplot(spec[1,3])
  plt.imshow(images['tvImgSample10'])
  plt.gray()
  plt.title('Kohinanpoisto, ' + str(angles) + ' kulmaa, 10 toistoa')
  plt.autoscale()

  fig.add_subplot(spec[1,2])
  plt.imshow(images['tvImgSampleMax'])
  plt.gray()
  plt.title('Kohinanpoisto, ' + str(angles) + ' kulmaa, ' + str(iterations) + ' toistoa')
  plt.autoscale()

  fig.set_size_inches(w=25,h=12)
  plt.autoscale()
