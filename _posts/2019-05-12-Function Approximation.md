### Introduce
---
- The **policy**, **value function** and **model** are all functions
- We want to learn (one of) these from experience 
- If there are too many states, we need to approximate
- In general, this is called RL with function approximation
- When using deep neural nets, this is often called deep reinforcement learning
- The term is fairly new --- the combination is decades old

<!--more-->
### Value Function Approximation
---
- **lookup tables**
  - Every state $s$ has an entry $q(s,a)$
  - Or every state-action pair $s,a$ has an entry $q(s,a)$
- Large MDPs:
  - There are too many states and/or actions to store in memory
  - It is too slow to learn the value of each state individually
  - Individual states are often **not fully observable**

- Solution for large MDPs:
  - Estimate value function with **function approximation**  
    $$
    \begin{aligned}
    v_\theta(s)\approx v_\pi(s) \qquad &(or, v_*(s)) \\
    q_\theta(s,a)\approx q_\pi(s,a) \qquad &(or,q_*(s,a))
    \end{aligned}
    $$  
  - **Generalise** from seen states to unseen states
  - **Update** parameter $\theta$ using MC or TD observable
- If the environement state is not fully observable:
  - Use the **agent state**
  - Consider learning a **state update function** $S_{t+1}=u(S_t,O_{t+1})$
  - Henceforth, $S_t$ denotes the agent state

### Which Function Approximator?
There are many function approximators, e.g.
  - Artificial neural network
  - Decision tree
  - Nearest neighbour
  - Fourier / wavelet bases
  - Coarse coding 

In principle, **any** function approximator can be used, but RL has specific properties:
  - Experience is not i.i.d --- successive time-step are correlated 
  - Agent's policy affects the data it receives
  - Value functions $v_\pi(s)$ can be non-stationary
  - Feedback is delayed, not instantaneous

### Classes of Function Approximation
---
- Tabular: a table with an entry for each MDP state
- State aggregation: Partition environment states
- Linear function approximate: fixed feature (or fixed kernel)
- Differentiable (nonlinear) function approximation: neural nets

### Approximate Values By Stochastic Gradient Descent
---
- Goal: fins $\theta$ that minimise the difference between $v_\theta(s)$ and   $v_\pi(s)$ 
$$
J(\theta)=\mathbb{E}[(v_\pi(S)-v_\theta(S))^2]
$$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Note: The expectation if over the state distribution --- e.g., induced by the policy
- Gradient descent:  
$$
\Delta\theta=-\frac{1}{2}\alpha\nabla_\theta J(\theta)=\alpha\mathbb{E}[(v_\pi(S)-v_\theta(S))\nabla_\theta v_\theta(S)]
$$
- **Stochastic** gradient descent  
$$
\Delta\theta_t=\alpha(v_\pi(S_t)-v_\theta(S_t))\nabla_\theta v_\theta(S_t)
$$
### Feature Vectors
---
- Represent state by a **feature vector**   
$$
\phi(s) = \left(\begin{array}{cc}
\phi_1(s) \\ \vdots \\ \phi_n(s)
\end{array}\right)
$$
- $\phi:S\rightarrow\mathbb{R}^n$ is a fixed mapping from state (e.g. observation) to features
- Short-hand: $\phi_t=\phi(S_t)$
- For example:
  - Distance of robot from landmarks
  - Trends in the stock market
  - Piece and pawn configurations in chess

### Approximate Values By Stochastic Gradient Descent
- Goal: fina $\theta$ that minimise the difference between $v_\theta(s)$ and $v_\pi(s)$ 
  $$
  J(\theta)=\mathbb{E}[(v_\pi(S)-v_\theta(S))^2]
  $$    
  Note: The expectation if over the state distribution --- e.g., induced by the policy.
- Gradient descent:  
$$
\Delta\theta=-\frac{1}{2}\alpha\Delta_\theta J(\theta)=\alpha\mathbb{E}_\pi p[(v_\pi(S)-v_\theta(S))\nabla_\theta v_\theta(S)]
$$
- **Stochastic** gradient descent:  
$$
\Delta\theta_t=\alpha(v_\pi(S_t)-v_\theta(S_t))\nabla_\theta v_\theta(S_t)
$$

### Linear Value Function Approximation
- Approximate value function by a linear combination of features  
$$
v_\theta(s)=\theta^\top \phi(s)=\sum_{j=1}^n\phi_j(s)\theta_j
$$
- Objective function ('loss') is quadratic in $\theta$  
$$
J(\theta)=\mathbb{E}_\pi\big[(v_\pi(S)-\theta^\top\phi(S))^2\big]
$$
- Stichastic gradient descent converges on global ooptimum
- Update rule is simple  
$$
\nabla_\theta v_\theta(S_t)=\phi(S_t)=\phi_t \quad \Longrightarrow \quad \Delta_\theta = \alpha(v_\pi(S_t)-v_\theta(S_t))\phi_t
$$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\large \text{Update}=\textbf{step size}\times\textbf{prediction error}\times\textbf{feature vector}$

### Incremental Prediction Algorithms
---
- The true value function $v_\pi(s)$ is typically not available
- In practice, we substitute a **target** for $v_\pi(s)$
  - For MC, the target is the return $G_t$  
$$
\Delta\theta_t=\alpha(G_t - v_\theta(s))\nabla_\theta v_\theta(s)
$$
  - For TD, the target is the TD target $R_{t+1}+\gamma v_\theta(S_{t+1})$  
  $$
  \Delta\theta_t=\alpha(R_{t+1}+\gamma v_\theta(S_{t+1}) - v_\theta(S_t))\nabla_\theta v_\theta(S_t)
  $$

### Monte-Carlo with Value Function Approximation
---
- The return $G_t$ is an unbiased, noisy sample of $v_\pi(s)$
- Can therefore apply supervised learning to (online) "training data"  
$$
{(S_0,G_0),\ldots,(S_t,G_t)}
$$
- For example, using **linear Monte-Carlo policy evaluation**  
  $$
  \begin{aligned}
  \Delta\theta_t &=\alpha(G_t - v_\theta(S_t))\nabla_\theta v_\theta(S_t) \\
  & = \alpha(G_t - v_\theta(S_t))\phi_t
  \end{aligned}
  $$
- Monte-Carlo evaluation converges to a local optimum
- Even when using non-linear value function approximation
- For linear function, it finds the globlal optimum

### TD Learning with Value Function Approximation
---
- The TD-target $R_{t+1}+\gamma v_\theta(S_{t+1})$ is a **biased** sample og true value $v_\pi(S_t)$
- Can still apply supervised learning to "training data"  
$$
{(S_0,R_1+\gamma v_\theta(S_1)),\ldots,(S_t,R_{t+1}+\gamma v_\theta(S_{t+1}))}
$$
- For example, using linear TD  
$$
\begin{aligned}
\Delta\theta_t &= \alpha\underbrace{(R_{t+1}+\gamma v_\theta(S_{t+1})-v_\theta(S_t))}_{\normalsize =\delta_t, \text{TD error}}\nabla_\theta v_\theta(S_t) \\
& =\alpha\delta_t\phi_t
\end{aligned}
$$

### Convergence of MC and TD
---
- with linear functions, MC converges to  
  $$
  \min_\theta\mathbb{E}\big[(G_t-v_\theta(S_t))^2\big]=\mathbb{E}\big[\phi_t\phi_t^\top\big]^{-1}\mathbb{E}\big[v_\pi(S_t)\phi_t\big]
  $$

- With linear function, TD converges to  
  $$
  \min_\theta\mathbb{E}\big[(R_{t+1}+\gamma v_\theta(S_{t+1}-v_\theta(S_t)))^2\big]=\mathbb{E}\big[\phi_t(\phi_t-\gamma\phi_{t+1})^\top\big]\mathbb{E}\big[R_{t+1}\phi_t\big]
  $$
  (in continuing problem with fixed $\gamma$)
- This is a different solution from MC
- Typically, the asymptotic MC solution is preferred
- But TD methods may converge faster,   and may still be better  
  $$
  \textbf{TD:}\quad\Delta_t=\alpha\delta\nabla_\theta v_\theta(S_t)\quad \textbf{where} \quad \delta_t=R_{t+1}+\gamma v_\theta(S_{t+1}-v_\theta(S_t))
  $$
- This update ignores dependence of $v_\theta(S_{t+1})$ on $\theta$

### Action-Value Function Approximation
- Approximate the action-value function  
$$
q_\theta(s,a)\approx q_\pi(S,a)
$$
- For instance, with linear function approximation  
$$
q_\theta(s,a)=\phi(s,a)_\top\theta=\sum_{j=1}^n\phi_j(s,a)\theta_j
$$
- Stochastic gradient descent update  
$$
\begin{aligned}
\Delta\theta &= \alpha(q_\pi(s,a)-q_\theta(s,a))\nabla_\theta q_\theta(s,a) \\
&= \alpha(q_\pi(s,a)-q_\theta(s,a))\phi(s,a)
\end{aligned}
$$

### Least Squarse Prediction
---
- Given value function approximation $v_\theta(s) \approx v_\pi(s)$
- And **experience** $\mathcal{D}$ consisting of $\large \langle \text{state, estimated value} \rangle$pairs  
$$
\mathcal{D}=\big\{\langle S_1,\hat{v}_1^\pi \rangle,\langle S_2,\hat{v}_2^\pi \rangle,\ldots,\langle S_T,\hat{v}_T^\pi \rangle \big\}
$$
- E.g., $\large \hat{V}_1^\pi=R_{t+1}+\gamma v_\theta(S_{t+1})$
- Which parameters $\theta$ give the best fitting value function $v_\theta(s)$?

### Stochastic Gradient Descent with Experience Replay
---
Give experience consisting of $\large \langle \text{state, value} \rangle$pairs  
$$
\mathcal{D}=\big\{\langle S_1,\hat{v}_1^\pi \rangle,\langle S_2,\hat{v}_2^\pi \rangle,\ldots,\langle S_T,\hat{v}_T^\pi \rangle \big\}
$$  
Repeat:
  1. Sample state, value from experience  
$$
\langle s, \hat{v}_\pi  \rangle \sim \mathcal{D}
$$
  2. Apply stochastic gradient decent update  
$$
\Delta\theta = \alpha(\hat{v}^\pi - v_\theta(s))\nabla_\theta v_\theta(s)
$$  
Converges to least squares solution  
$$
\theta_\pi=\mathop{\text{argmin}}\limits_\theta LS(\theta)=\mathop{\text{argmin}}\limits_\theta\mathbb{E}_\mathcal{D}\big[(\hat{v}_i^\pi-v_\theta(S_i))^2\big]
$$

### Linear Least Squares Prediction
---
- Experience replay finds least squares solution
- But it may take many iterations
- Using **linear** value function approximation $v_\theta(s)=\phi(s)^\top\theta$ we can solve the least squares solution directly
- At minimum of $LS(\theta)$, the expected update must be zero  
  $$
  \begin{aligned}
  \mathbb{E}_\mathcal{D}[\Delta\theta] &= 0 \\
  \alpha\sum_{t=1}^T\phi_t(\hat{v}_t^\pi-\phi_t^\top\theta) &= 0 \\
  \sum_{t=1}^T\phi_t\hat{v}_t^\pi &= \sum_{t=1}^T\phi_t\phi_t^\top\theta \\
  \theta_t &= \Big(\sum_{t=1}^T\phi_t\phi_t^\top\Big)^{-1}\sum_{t=1}^T\phi_t\hat{v}_t^\pi
  \end{aligned}
  $$
- For N feature, direct solution time is $O(N^3)$
- Incremental solution time is $O(N^2)$ using Shermann-Morrison
- We do not know true values $v_\pi$ (have estimates $\hat{v}_t$)
- In practice, our "training data" must use noisy or biased sample of $v_\pi$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <font color=blue>**LSMC**</font> Least Squares Monte-Carlo uses return  
$$
v_\pi \approx G_t
$$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <font color=blue>**LSTD**</font> Least Squares Temporal-Difference uses TD target  
$$
v_\pi \approx R_{t+1} + \gamma v_\theta(S_{t+1})
$$
- In each case we can solve directly for the fixed poine

### Deep reinforcement learning
---
- Many ideas immediately transfer when using deep neural networks:
  - TD and MC
  - Double learning (e.g., double Q-learning)
  - Experience replay
  - ...
- Some ideas do not easily transfer
  - UCB
  - Least squares TD/MC
 
### Neural Q-learning
---
- Online neural Q-learning may include:
  + A **network** $q_\theta:\space O_t \Longrightarrow (q[1],\ldots,q[m])(m\space \text{actions})$
  + An $\epsilon-\text{greedy}$ **exploration policy**: $q_t\space \Longrightarrow \space \pi_t \Longrightarrow \space A_t$
  + A Q-learning **loss function** on $\theta$  
    $$
    I(\theta)=\frac{1}{2}\Big(R_{t+1}+\gamma\Big[\max_a q_\theta (S_{t+1},a)\Big] - q_\theta (S_t, A_t)\Big)^2
    $$  
    where $[\cdot ]$ denotes stopping the gradient, so that the gradient is  
    $$
    \nabla_\theta I(\theta)=\Big(R_{t+1}+\gamma\max_a q_\theta(S_{t+1},a)-q_\theta(S_t,A_t)\Big)\nabla_\theta q_\theta(S_t,A_t)
    $$
  + An **optimizer** to minimize the loss (e.g., SGD, RMSProp, Adma)

### DQN
--- 

- DQN (Mnih et al. 2013, 2015) includes;
  + A **network** $q_\theta:\space O_t \mapsto (q[1],\ldots,q[m])(m\space \text{actions})$
  + An $\epsilon-\text{greedy}$ **exploration policy**: $q_t\space \mapsto \space \pi_t \Longrightarrow \space A_t$
  + A **replay buffer** to store and sample past transitions
  + A **target network** $q_{\theta^-}:\space Q_t \mapsto\space(q^-[1],\ldots,q^-[m])$
  + A Q-learning **loss function** on $\theta$ (use replay and target network)   
    $$
    I(\theta)=\frac{1}{2}\Big(R_{t+1}+\gamma\Big[\max_a q_{\theta^-} (S_{t+1},a)\Big] - q_\theta (S_t, A_t)\Big)^2
    $$
  + An **optimizer** to minimize the loss (e.g., SGD, RMSProp, Adma)
- Replay and target networks make RL look more like supervised learning
- It is unclear whether they are vital, but they help
- "DL-aware RL"

### n-Step Return
---
- Consider the following n-steps returns for $n=1,2,\infty: $  
  $$
  \begin{aligned}
  &n=1 \quad &(TD)\quad & G_T^{(1)}=R_{t+1} + \gamma v(S_{t+1}) \\
  &n=2 \quad &\quad & G_T^{(2)}=R_{t+1} + \gamma R_{t+2} +\gamma^2 v(S_{t+2}) \\
  &\quad \vdots \quad &\quad & \vdots \\
  &n=\infty \quad &(MC)\quad & G_T^{(\infty)}=R_{t+1} + \gamma R_{t+2}+\ldots+\gamma^{T-t-1}R_T
  \end{aligned}
  $$

- Define the n-step return  
$$
G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \ldots + \gamma^{n-1} R_{t+n} + \gamma^n v(S_{t+n})
$$
- n-step temporal-difference learning  
$$
v(S_t) \leftarrow v(S_t) + \alpha\Big(G_t^{(n)} - v(S_t)\Big)
$$