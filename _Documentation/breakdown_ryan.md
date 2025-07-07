# [Infinite GAN Breakdown](https://arxiv.org/abs/2102.03657)

> Author: Ryan Roby

Infinite GAN uses a GAN framework to model stochastic signals with an arbitrary number of timesteps and dimensions (hence "infinite dimensions"), where the dynamics are governed by a stochastic differential equation (SDE) defined as:

$dX_t =f(t,X_t) + g(t,X_t)\circ dW_t$

Where $X_t$ is a continuous $\mathbb{R}^x$ -valued stochastic process, $f[0,T] \times \mathbb{R}^x \rightarrow\mathbb{R}^x$, $g[0,T] \times \mathbb{R}^x \rightarrow\mathbb{R}^{x\times w}$ and $W_t$ is a $w$-dimensional Brownian motion. The operator $\circ$ is the Stratonovich Integral.

# Model Description

## 1. Generator

The generator converts an initial gaussian noise $V$ and Brownian Motion $W$ to a signal whose dynamics is similar to our target stochastic process.The initial noise is independent of the Brownian Motion.

Neural SDE the generator tries to model are of the form:

##### Governing Equations

$X_0=\zeta_\theta (V)$
$dX_t = \mu_\theta(t,X_t)dt + \sigma_\theta(t,X_t)\circ dW_t$
$Y_t=\alpha_\theta X_t + \beta_\theta$

##### Notations

$t\in[0,T]$: The time states
$X_t \in \mathbb{R}^x$: Hidden layer signal value at the $t$-th time state.
$Y_t\in\mathbb{R}^d$: Generated signal value at the $t$-th time state.
$W_t\in\mathbb{R}^w$: Brownian motion at the $t$-th time state.
$V \in\mathbb{R}^v$: Initial Gaussian noise.

### 1.1 Implementation in Code

#### Components:

1. **Initial layer ($\zeta_\theta$)**
   A neural network of with Lipswish activation after each linear layer except the last one which has no activation function.
   $\quad \zeta_\theta : \mathbb{R}^v \rightarrow \mathbb{R}^x$
2. **Drift process ($\mu_\theta$)**
   A neural network of with Lipswish activation after each linear layer. The last activation is a tanh activation function.
   $\quad \mu_\theta : \mathbb{R}^{h+1} \rightarrow \mathbb{R}^x$
3. **Diffusion process ($\sigma_\theta$)**
   A neural network of with Lipswish activation after each linear layer. The last activation is a tanh activation function.
   $\quad \sigma_\theta : \mathbb{R}^{x+1} \rightarrow \mathbb{R}^{x\times w}$
4. **Readout layer ($\alpha_\theta, \beta_\theta$)**
   A linear layer which converts it from the signals in the hidden layer to real signal dimensions. Let us combine $\alpha_\theta$ and $\beta_\theta$, and represent it as $\gamma_\theta$.
   $\quad \gamma_\theta : \mathbb{R}^{x} \rightarrow \mathbb{R}^{d}$

#### Process breakdown:

1.  Sample an initial noise using the noise hyper parameters and pass it through the initial layer to get $X_0$.

        init_noise = torch.randn(batch_size, self._initial_noise_size, device=ts.device)
        x0 = self._initial(init_noise)

2.  Compute the $X_t$ for the next states using the SDE adjoint solver using the _reversible heun_ method.

        xs = torchsde.sdeint_adjoint(self._func, x0, ts, method='reversible_heun', dt=1.0,
                                     adjoint_method='adjoint_reversible_heun',)
        xs = xs.transpose(0, 1)

3.  Pass the signals in the hidden space $X_t$ through the readout layer.

        ys = self._readout(xs)

4.  Return the linearly interpolatable version of the signal timestamps and values.

        ts = ts.unsqueeze(0).unsqueeze(-1).expand(batch_size, ts.size(0), 1)
        return torchcde.linear_interpolation_coeffs(torch.cat([ts, ys], dim=2))

## 2. Discriminator

The discriminator takes in a signal(generated or real) and returns a score that is obtained from the terminal state of an embedded signal $H_T$ in the hidden space.

Neural SDE the generator tries to model are of the form:

##### Governing Equations

$H_0=\xi_\phi (Y_0)$
$dH_t = f_\phi(t,H_t)dt + g_\phi(t,H_t)\circ dY_t$
$D=m_\phi . H_T$

##### Notations

$t\in[0,T]$: The time states
$Y_0 \in \mathbb{R}^d$: Initial state of the signal.
$H_0 \in \mathbb{R}^h$: Initial state of the signal in the hidden space.
$Y_t\in\mathbb{R}^d$: Signal value at the $t$-th time state.
$H_t\in\mathbb{R}^h$: Signal value at the $t$-th time state in the hidden space.
$H_T\in\mathbb{R}^h$: Terminal state of the signal in the hidden space.
$D \in\mathbb{R}^1$: Score value.

### 2.1 Implementation in Code

#### Components:

1. **Initial layer ($\xi_\phi$)**
   A neural network of with Lipswish activation after each linear layer except the last one which has no activation function.
   $\xi_\phi : \mathbb{R}^y \rightarrow \mathbb{R}^h$
2. **Module\Diffusion process ($g_\phi$)**
   A neural network of with Lipswish activation after each linear layer. The last activation is a tanh activation function.
   $g_\phi : \mathbb{R}^{h+1} \rightarrow \mathbb{R}^{h\times(d+1)}$
3. **Readout layer ($m_\phi$)**
   A linear layer which converts it from the hidden dimensions to a 1D score.
   $m_\phi : \mathbb{R}^h \rightarrow \mathbb{R}^1$
    > **Drift process ($f_\phi$)** has not been defined. It gets defined later down the line but its parameters are set to zero.

#### Process breakdown:

1.  The discriminator takes in an interpolatable version of the signal $Y$ using the linear interpolation function in _torchcde_.

        torchcde.LinearInterpolation(real_samples)

2.  Then the initial values of the signals are extracted and passed into the initial neural network, $\xi_\phi$, to get $H_0$

        Y0 =Y.evaluate(Y.interval[0])
        h0 = self._initial(Y0)

3.  Computes the terminal state of the signals in the hidden space, $H_T$ using a controlled differential equation solver in the SDE setting.

> **Info**: It initially converts the interpolatable signal $Y$ into a vector field $K$. Then passes it into a adjoint sde solver. The solver uses _reversible heun_ method to solve the SDE.
>
> When the interpolatable signal $Y$ is converted into a vector field, it initialises the diffusion term $q$ as a zero-like with the same dimension as the vector field and sets the drift term $s=g_\phi\times\frac {dY}{dt}$. Once it passes to the SDE solver, the drift term $s$ essentially becomes the diffusion term from the governing equations of discriminator model, but $f_\phi$ = 0.

> **Doubt**: Since the diffusion term of the vector field is set to zero, would it mean that the drift term in the governing equation $f_\phi$ is zero

    hs = cdeint_test(Y, func, h0, Y.interval, method='reversible_heun', backend='torchsde', dt=1.0,
                        adjoint_method='adjoint_reversible_heun',
                        adjoint_params=(real_samples,) + tuple(func.parameters()))

4. The terminal states, $H_T$ are passed to a linear layer $m_\phi$ to get a final score. The scores are then averaged across the batch.

> **Info**: Given signals of similar dynamics ($dY$) and starting point($Y_0$), the eventual terminal point obtained through SDE integration must be similar. I believe this is the logic behind the scoring function.

    score = readout(hs[:, -1])

## 3. Training

The objective function for the generator is:

$\min_\theta [E_{V,W} D_{\phi}(Y_\theta(V,W))]$

and that of the discriminator is

$\max_\phi [E_{V,W}D_{\phi}(Y_\theta(V,W))-E_zD_{\phi}(\hat z)]$

Where $\hat z$ are real signals.

        generated_samples = generator(ts, batch_size)
        generated_score = discriminator(generated_samples)
        real_score = discriminator(real_samples)
        loss = generated_score - real_score
        loss.backward()

        for param in generator.parameters():
            param.grad *= -1
        generator_optimiser.step()
        discriminator_optimiser.step()
        generator_optimiser.zero_grad()
        discriminator_optimiser.zero_grad()

### Lipschitz regularisation

Lipschitz regularisation is performed for the discriminator parmeters at this block:

        with torch.no_grad():
            for module in discriminator.modules():
                if isinstance(module, torch.nn.Linear):
                    lim = 1 / module.out_features
                    module.weight.clamp_(-lim, lim)

### Stochastic Weight Averaging (SWA)

SWA is performed for both the generator and discriminator for added stability.

    averaged_generator = swa_utils.AveragedModel(generator)
    averaged_discriminator = swa_utils.AveragedModel(discriminator)

### For specific tasks

#### 1. Classification

A neural CDE was used as a classifier. Larger loss leads to better performance for the generative model.

#### 2. Prediction

Sequence-to-sequence
model to predict the latter part of a time series given the first part, using generated data. Testing is performed on real data. They used a neural CDE/ODE as an encoder/decoder pair. Smaller losses, meaning ability to predict, are better.

#### 3. Maximum mean discrepancy

Distance between probability distributions with respect to a kernel or feature map. They used the depth-5 signature transform as the feature map.Smaller values, meaning closer distributions, are better.

> **Doubt**: Unclear as to how the loss from these tasks are fed to the generator or discriminator loss (regular summation or weighted summation)
