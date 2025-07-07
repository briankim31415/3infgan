# Problem Statement

This project investigates **Infinite GANs** for generating time-varying urban scene data such as mobility traces, traffic simulations, or sensor time series.

## 🧠 Why Infinite-Dimensional GANs?

Traditional GANs generate data in **fixed-dimensional space** (e.g., 128×128 pixel RGB images as points in $\mathbb{R}^{128 \times 128 \times 3}$). Each sample is a **static point**.

However, urban scene data is often **dynamic and continuous**:

-   Trajectories of vehicles or pedestrians
-   Sequences of street-view frames
-   Environmental sensor readings over time

These are better modeled as **functions over time**, not static vectors.

**Infinite GANs** address this by defining the generator as a stochastic process — it outputs full _trajectories_, not just single data points. The result is more realistic, temporally coherent synthetic data.

## 🔍 Core Modeling Idea

Let $\mathcal{Z}$ be the space of continuous time series (e.g., paths in an urban environment). We define:

-   A **Neural SDE Generator**:

    ```math
    dX_t =f(t,X_t) + g(t,X_t)\circ dW_t, \quad X_0 \sim \mu
    ```

    -   $f$ and $g$ are neural networks
    -   $W_t$ is Brownian motion
    -   $X_t$ evolves over time and produces an output path $z = \phi(X_{\cdot})$
    -   $\mu$ is the initial probability distribution

-   A **Neural CDE Discriminator**:
    ```math
    H_0=\xi_\phi (Y_0), \quad dH_t = f_\phi(t,H_t)dt + g_\phi(t,H_t)\circ dY_t, \quad D=m_\phi . H_T
    ```
    -   $\xi_\phi$, $f_\phi$, and $g_\phi$ are neural networks
    -   $H$ is the "well-behaved" path space
    -   $Y$ is the generator's output
    -   $D$ is the discriminator's output
    -   This lets the discriminator evaluate _entire_ trajectories, not just individual samples

The model is trained using a **Wasserstein GAN objective**, minimizing the difference between the discriminator's scores for the real and fake samples.

## 🎯 Goal

Learn a generator such that the distribution of generated trajectories closely matches the real-world data distribution in terms of path-level behavior, not just pointwise similarity.
