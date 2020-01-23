### Introduction to MDPs
---
1. **Markov decision processes (MDPs)** formally describe an environment.
2. Assume the environment is fully observable the current: the current observations contains relevant information.  
3. Almost all RL problems can be formalized as MDPs, e.g,
- Optimal control primarily deals with continuous MDPs
- Partially observable problems can be converted into MDPs
- Bandits are MDPs with one state

<!--more-->
---
### Definition
A state $s$ has the **Markvo** property when for states $\forall s^{'}\in S$ and all rewards $r \in \mathbb{R}$ 
$$
p(R_{t+1}=r,S_{t+1}=s^{'}|S_{t}=s)= 
p(R_{t+1}=r,S_{t+1}=s^{'}|S_{1},...,S_{t},S_{t-1}=s)
$$  
for all possible histories $S_{1},...,S_{t-1}$

- The state capture all relevent information from history
- Once the state is know, the history may be throw away
- The state is a sufficient statistic of the past


### Return 
---
+ Acting in a MDP results in **return** $G_{t}$: total discounted reward from time-step $t$  
    $$
    G_{t}=R_{t+1}+\gamma R_{t+2}+ ... = \sum_{k=0}^{\infty}\gamma_{k}R_{t+k+1}
    $$

+ This is a random variables that depends on **MDP** and **policy** 
+ The **discount** $\gamma\in[0,1]$ is the present value of future rewards
    + The marginal value of receiving reward $R$ after $k+1$time-steps is $\gamma^{k}R$
    + For $\gamma<1$, immediate rewards are more important than delayed rewards 
    + $\gamma$ close to 0 leads to "myopic" evaluation
    + $\gamma$ close to 1 leads to "far-sighted" evaluation

### Value Function
---
- The value function $v(s)$ gives the long-term value of state $s$   
    $$
    v_{\pi}(s)=\mathbb{E}[G_{t}|S_{t}=s,\pi]
    $$
- It can be defined recursively  
    $$
    \begin{aligned}
    v_{\pi}(s) &=\mathbb{E}[R_{t+1}+\gamma G_{t+1}|S_{t}, \pi] \\
    &=\mathbb{E}[R_{t+1}+\gamma v_{\pi}(S_{t+1})|S_{t}=s, A_{t}\sim\pi(S_{t})] \\
    &=\sum_{a}\pi(a|s)\sum_{r}\sum_{s^{'}}p(r,s^{'}|s,a)(r+\gamma v_{\pi}(s^{'}))
    \end{aligned}
    $$
- The final step writes out the expectation explicitly 

### Action Values
---
- We can define state-action values    
    $$
    q_{\pi}(s,a)=\mathbb{E}[G_{t}|S_{t}=s,A_{t}=a,\pi]
    $$
- This implies: the value of a state is equal to the weighted sum of the state action value by definition  
    $$
    \begin{aligned}
    q_{\pi}(s,a)&=\mathbb{E}[R_{t+1}+\gamma v_{\pi}(S_{t+1})|S_{t}=s, A_{t}=a] \\
    &=\mathbb{E}[R_{t+1}+\gamma q_{\pi}(S_{t+1},A_{t+1})|S_{t}=s,A_{t}=a] \\
    &=\sum_{r}\sum_{s^{'}}p(r,s^{'}|s,a)\Big(r+\gamma\sum_{a^{'}}\pi(a^{'},s^{'})q_{\pi}(s^{'},a^{'})\Big)
    \end{aligned}
    $$  

- Note that   
    $$
    \begin{aligned}
    v_{\pi}(s) & =  \sum_{a}\pi(a|s)q_{\pi}(s,a) \\ 
    & = \mathbb{E}[q_{\pi}(S_{t},A_{t})|S_{t}=s,\pi], \forall s
    \end{aligned}
    $$

---
#### Tips

- Estimating $v_{\pi}$ or $q_{\pi}$ is called **policy evaluation** or, simply, **prediction**
- Estimating $v_{\star}\space \text{or}\space q_{\star}$ is sometimes called **control**, because these can be used for **policy optimizaton**.

---
### Bellman Equation
Four Bellman equations:  
$$
\begin{aligned}
v_{\pi}(s) &=\mathbb{E}[R_{t+1}+\gamma v_{\pi}(S_{t+1})|S_{t}=s,A_{t}\sim\pi(S_{t})] \\
v_{*}(s) &=\max_{a}\mathbb{E}[R_{t+1}+\gamma v_{*}(S_{t+1})|S_{t}=s,A_{t}=a] \\
q_{\pi}(s,a) &=\mathbb{E}[R_{t+1}+\gamma q_{\pi}(S_{t+1}, A_{t+1})|S_{t}=s,A_{t}=a] \\
q_{*}(s,a) &=\mathbb{E}[R_{t+1}+\gamma \max_{a^{'}} q_{*}(S_{t+1},a^{'})|S_{t}=s,A_{t}=a]
\end{aligned}
$$
 
 ---
 ### Policy Evaluate
 
 - We start by discussing how to estimate  
    $$
    v_{\pi}(s)=\mathbb[R_{t+1}=\gamma v_{\pi}(S_{t+1})|s,\pi]
    $$
 - Idea: turn this equality into an update
 - First, initialize $v_{0}$ e.g. to zero.
 - Then iterate  
    $$
    \forall s: v_{k+1}(s)=\mathbb{E}[R_{t+1}+\gamma v_{k}(S_{t+1})|s,\pi]
    $$
- Note: whenever $v_{k+1}(s)=v_{k}(s)$, for all $s$, we must have found $v_{\pi}$

- This policy evaluation is always converge under appropriate conditions (e.g., $\gamma < 1$)

Implies $\lim_{k\rightarrow\infty}v_{k}=v_{\pi}$
- Finite-horizon episodic case is a bit harder, but also works





