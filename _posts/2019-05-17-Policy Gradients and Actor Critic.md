### General Overview
- Model-based RL:  
   \+ 'Easy' to learn a model (supervised learning)   
   \+ Learns 'all there is to know' from the data  
   \- Objective capture irrelevant information  
   \- May focus compute/capacity on irrelevant detail  
   \- Computing policy (planning) is non-trivial and can be computationally expensive
- Valued-based RL:  
   \+ Closer to true objective
   \+ Fairly well-understood --- somewhat similar to regression  
   \- Still not the true objective --- may still focus capacity on less-important details 
- Policy-based RL:  
   \+ Right objective!
   \- Ignores other learnable knowledge (potentially not the most efficient use of data) 

<!--more-->
---
### Advantage of Policy-Based RL
Advantages:
  - Good convergence properties
  - Easily extended to high-dimensional or continuous action spaces 
  - Can learn **stochastic** policies 
  - Sometimes Policies are **simple** while values and models are complex
    - E.g., rich domain, but optimal is always go left. 

Disadvantages:
  - Susceptible to local optimal (expecially with non-linear FA)
  - Obtained knowledge is specific, does not always generalize well 
  - Ignores a lot of information in the data (when used in isolation)

---
### Policy Objective Functions
- Goal: given policy $\pi_{\theta}(s,a)$ with patameters $\theta$, find best $\theta$
- But how do we measure the quality of a policy $\pi_{\theta}$?
- In episodic environments we can use the **start value**  
$$
J_{1}(\theta)=v_{\pi_{\theta}}(s_{1})
$$  
- In continuing environments we can use the average value  
$$
J_{avV}(\theta)=\sum_{s}\mu_{\pi_{\theta}}(s)v_{\pi_{\theta}}(s)
$$  
  where $\mu_{\pi}(s)=p(S_{t}=s|\pi)$ is the probability of being in state s in the long run
  Think of is as the ratio of time spent in $s$ under policy $\pi$
- Or the average reward per time-step  
$$
J_{avR}(\theta)=\sum_{s}\mu_{\pi_{\theta}}(s)\sum_{a}\pi_{\theta}(s,a)\sum_{r}p(r|s,a)r
$$
 
---
### Gradients on Parameterized Policies
- We need to compute an estimate of the policy gradient 
- Assume policy $\pi_{\theta}$ is differentiable almost everywhere 
    - E.g., $\pi_{\theta}$is a linear function of the agent state, or a neural network
    - Or we could have a parameterized class of controllers
- Goal is to compute  
$$
\nabla_{\theta}J(\theta)=\nabla\mathbb{E}_{d}[v_{\pi_{\theta}}(s)]
$$
- We will use Monte Carlo samples to compute this gradient
- So, how does $\mathbb{E}_{d}[v_{\pi_{\theta}}(S)]$ depend on $\theta$?

---
### Policy Gradient Theorem

- The policy gradient approach also applies to (multi-step) MDPs
- Replaces instantaneous reward R with long-term value $q_{\pi}(s,a)$
- Policy gradient theorem applies to start state objective, average reward and average value objective

**Theorem**
For any differentiable policy $\pi_{\theta}(s,a)$, for any of the policy objective functions $J=J_{1}, J_{avR}, or 1\frac{1}{1-\gamma}J_{avV}$, the policy gradient is  
$$
\nabla_{\theta}J(\theta) = \mathbb{E}[q_{\pi_{\theta}}(S,A)\nabla_{\theta}\log\pi_{\theta}(A|S)]
$$  
Expectation is over both states and actions

---
### Policy Gradients on trajectories: Derivation 
- Consider trajectory $\zeta=S_0,A_0,R_0,S_1,A_1,R_1,S_2,\ldots$ with return $G(\zeta)$  
$$
\begin{aligned}
& \nabla_\theta J_\theta(\pi)=\nabla_\theta\mathbb{E}[G(\zeta)]=\mathbb{E}[G(\zeta)\nabla_\theta\log p(\zeta)]\qquad \text{(score function trick)} \\
& \nabla_\theta\log p(\zeta) \\
& =\nabla_\theta\log\Big[p(S_0)\pi(A_0|S_0)p(S_1|S_0,A_0)\pi(A_1|S_1)\cdots)\Big] \\
& =\nabla_\theta\Big[\log p(S_0)+\log\pi(A_0|S_0)+\log p(S_1|S_0,A_0)+\log\pi(A_1|S_1)+\cdots)\Big] \\
& =\nabla_\theta\Big[\log\pi(A_0|S_0)+\log\pi(A_1|S_1)+\cdots\Big] \\
\textbf{So:} \\
& \nabla_\theta J_\theta(\pi)=\mathbb{E}\Big[G(\zeta)\nabla_\theta\sum_{t=0}\log\pi(A _t|S_t)\Big] = \mathbb{E}\Big[\Big(\sum_{t=0}R_{t+1}\Big)\Big(\nabla_\theta\sum_{t=0}\log\pi(A_t|S_t)\Big)\Big]
\end{aligned}
$$

---
### Policy gradients on trajectories: reduce variance
- Note that, in general   
$$
\begin{aligned}
\mathbb{E}[b\nabla_{\theta}\log\pi(A_{t}|S_{t})]&=\mathbb{E}\Bigg[\sum_{a}\pi(a|S_{t})b\nabla_{\theta}\log\pi(a|S_{t})\Bigg] \\
&=\mathbb{E}\Bigg[b\nabla_{\theta}\sum_{a}\pi(a|S_{t})\Bigg] \\
&=\mathbb{b}[b\nabla_{\theta}1] \\
&=0
\end{aligned}
$$
- The sum of probability distribution is 1
- This holds only if $b$ does not depend on the action (though it can depend on the state)
- Implies we can subtract a **baseline** to reduce variance

- Consider trajactory $\zeta=S_{0},A_{0},R_{0},S_{1},A_{1},R_{1},S_{2},\ldots$ with return $G{\zeta}$  
  $$
  \nabla_{\theta}J_{\theta}(\pi)=\mathbb{E}\Bigg[\Bigg(\sum_{t=0}R_{t+1}\Bigg)\Bigg(\nabla_{\theta}\sum_{t=0}log\pi(A_{t}|S_{t})\Bigg)\Bigg]
  $$    
  but $\sum_{t=0}^{k}R_{t+1}$ does not depend on actions $A_{k+1}, A_{k+2},\cdots,$so  
  $$
  \begin{aligned}
  &=\mathbb{E}\Bigg[\sum_{t=0}\nabla_{\theta}\log\pi(A_{t}|S_{t})\sum_{i=0}R_{i+1}\Bigg] \\
  &=\mathbb{E}\Bigg[\sum_{t=0}\nabla_{\theta}\log\pi(A_{t}|S_{t})\sum_{i=\color{red}{t}}R_{i+1}\Bigg] \\
  &=\mathbb{E}\Bigg[\sum_{t=0}\nabla_{\theta}\log\pi(A_{t}|S_{t})q_{\pi}(S_{t},A_{t})\Bigg]
  \end{aligned}
  $$  

- A good baseline is $v_{\pi}(S_{t})$  
$$
\nabla_{\theta}J_{\theta}(\pi)=\mathbb{E}\Bigg[\sum_{t=0}\nabla_{\theta}\log\pi(A_{t}|S_{t})q_{\pi}(S_{t},A_{t})-v_{\pi}(S_{t})\Bigg]
$$

- Typically, we estimate $v_{w}(s)$ explicitly, and sample  
$$
q_{\pi}(S_{t},A_{t})\approx G_{t}^{(n)}
$$
- For instance, $G_{t}^{(1)}=R_{t+1}+ \gamma v_{w}(S_{s+1})$ 

---
### Estimating the Action-Value Function
- The Critic is solving a familiar problem: policy evaluation
- What is the value of policy $\pi_{\theta}$ for current parameters $\pi$  

---
### Actor-Critic
**Critic** Update parameters $w$ of $v_{w}$ by n-step TD (e.g., $n=1$)
**Actor** Update $\theta$ by policy gradient  

$$
\begin{aligned}
& \textbf{function}\space\text{ADVANTAGE ACTOR CRITIC} \\
& \qquad\text{Initialise}\quad s, \theta \\
& \qquad\textbf{for}\quad t=0,1,2,\ldots\space \textbf{do} \\
& \qquad\qquad \text{Sample}\space A_t \sim \pi_\theta(S_t) \\
& \qquad\qquad \text{Sample}\space R_{t+1}\space\text{and}\space S_{t+1} \\
& \qquad\qquad \delta_t=R_{t+1}+\gamma v_w(S_{t+1})-v_w(S_t) \qquad \text{[one-step TD-error, or \color{red}{advantage}]}\\
& \qquad\qquad w\leftarrow w+\beta\delta_t\nabla_wv_w(S_t) \qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\quad \text{[TD(0)]} \\
& \qquad\qquad \theta\leftarrow\theta+\alpha\delta_t\nabla_\theta\log\pi_\theta(A_t|S_t) \qquad\qquad\qquad\quad\space\space\thinspace\text{[Policy gradient update]}\\
& \qquad\textbf{end for} \\
& \textbf{end function}
\end{aligned}
$$

---
### Full Advantage Actor Critic Agent
- Adventage actor critic include
  - A **representation** (e.g., LSTM): $(S_{t-1}, O_t)\rightarrow S_t$
  - A **network** $v_w: \space S\rightarrow v$
  - A **network** $\pi_\theta:\space S\rightarrow \pi$
  - Copies/variants $\pi^m$ of $\pi_\theta$ to use as **policies**: $S_t^m\rightarrow A_t^m$
  - A n-step TD **loss** on $v_w$  
    $$
    I(w)=\frac{1}{2}\Big(G_t^{(n)}-v_w(S_t)\Big)^2
    $$  
    where $G_t^{(n)} = R_{t+1}+\gamma R_{t+2}+\ldots+\gamma^{n-1}v_w(S_{t+n})$
  - A n-step REINFORCE **loss** on $\pi_\theta$  
  $$
  I(\theta)=\Big[G_t^{(n)}-v_w(S_t)\Big]\log\pi_\theta(A_t|S_t)
  $$
  - **Optimizers** to minimize the losses
- Also know as A2C, or A3C (when combined with asynchronous parameter update)

---
### Bias in Actor-Critic Algorithm
- Approximating the policy gradient introduce bias 
- A biased policy gradient may not find the right solution
- Full return: high variance
- One-step TD-error: high bias
- n-step TD-error: useful middle ground  
$$
\begin{aligned}
\delta_t^{(n)} &= G_t^{(n)} - v_w(S_t) \\
&=\underbrace{R_{t+1} + \gamma R_{t+2}+ \cdots + \gamma^{n-1}R_{t+n} + \gamma^{n}v_w(S_t+n)}_{=G_t^{(n)}}-v_w(S_t)
\end{aligned} 
$$
- It is really important to use close-to on-policy targets
- If needed, use importance sampling to correct  
  $$
  G_t^{(n),\rho} = \frac{\pi_\theta(A_t|S_t)}{b(A_t|S_t)}\Big(R_{t+1}=\gamma_{t+1}^{(n-1),\rho}\Big)
  $$  
  with  
  $$
  G_t^{(0),\rho} =v_w(S_t) \approx v_\pi(S_t)
  $$

---
#### $\Large \color{dark}{\lambda}\textbf{-returns}$
- We can write a multi-step return recursively  
$$
\begin{aligned}
& G_t^{(n)}=R_{t+1}+\gamma G_{t+1}^{n-1} \\
& G_t^{(0)} = v_w(S_t) \approx v_\pi(S_t)
\end{aligned}
$$
- This is equivalent to   
$$
G_t^\lambda = R_{t+1} + \gamma(1- \lambda_{t+1})v_w(S_{t+1}) + \gamma\lambda_{t+1}G_{t+1}^\lambda
$$
- We can generalize to $\lambda_t \in [0,1]$; this is called a $\lambda-\text{return}$
- It can be interpreted as a **mixture of n-step returns**
- One way to correct for off-policy returns: bootstrap (set $\lambda=0$) whenever the policies differ
- Can be used for policy-gradient and value prediction

---
### Trust Region Policy Optimization
- Many extensions and variants exist
- Important: be careful with update: a bad policy leads to bad data
- This is different from supervised learning (where learning and data are independent)
- One solution: regularise policy to not change too much

---
### Increasing Robustness with Trust Regions
- One way to prevent instability is to **regularise**
- A popular method is to **limit the difference between subsequent policies**
- For instance, use the Kullbeck-Leibler divergence:  
  $$
  KL(\pi_{old}\|\pi_\theta)=\mathbb{E}\Big[\int\pi_{old}(a|S)\log\frac{\pi_\theta(a|S)}{\pi_{old}(a|S)}da\Big]
  $$  
(a divergence is like a distance --- but between distributions)
- Then maximise $J(\theta)-\eta KL(\pi_{old}\|\pi_\theta)$, for some small $\eta$
- It can also help to use large batches \
 c.f **TRPO** (Schulman et al. 2015) and **PPO** (Abbeel & Schulman 2016)

---
### Gaussian POlicy
- In conitnuous action spaces, a Gaussian policy is common
- E.g., mean is some function of state $\mu(s)$
- For simplicity, lets consider fixed variance of $\sigma^2$ (can be parameterized as well, instead) 
- Policy is Gaussian, $a \sim \mathcal{N}(\mu(s),\sigma^2)$
- The gradient of the log of the policy is then  
$$
\nabla_\theta\log\pi_\theta(s,a)=\frac{a-\mu(s)}{\sigma_2}\nabla\mu(s)
$$
- This can be used, for instance, in REINFORCE / advandage actor critic

---
### Continuous Actor-Critic Learning Automaton （Cacla） 

$$
\begin{aligned}
& \blacktriangleright a_t=Actor_\theta(S_t)\qquad  &\text{(get current (continuous) action proposal)}&\\
& \blacktriangleright A_t\sim\pi(\cdot|S_t,a_t)\space(e.g.,A_t \sim \mathcal{N}P(a_t, \sum) \qquad &\text{(explore)}&\\
& \blacktriangleright \delta_t=R_{t+1}+\gamma v_w(S_{t})\qquad &\text{(compute TD error)} &\\
& \blacktriangleright \text{Update}\space v_w{S_t}\space(\text{e.g.,using TD}) \qquad&\text{(policy evaluation)} &\\
& \blacktriangleright \text{If}\space \delta_t > 0, \text{update Actor}_\theta(S_t)\space\text{towards}\space A_t \qquad&\text{(policy improvement)}&\\
& \blacktriangleright \text{if}\space \delta_t \leqslant 0, \text{do not update Actor}_\theta \qquad&\text{} & 
\end{aligned}
$$
