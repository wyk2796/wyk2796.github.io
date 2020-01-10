### 1. Model-Based and Model-Free RL
- Model-Free RL
  - No model 
  - **Learn** value function (and/or policy) from experience
- Model-Based RL
  - **Learn** a model from experience OR be given a model
  - **Plan** value function (and/or policy) from model

<!--more-->

---
### 2. What is a Model?
- A model $\mathcal{M}_\eta$ is a representation of MDP $\langle \mathcal{S, A}, \hat{p}_\eta \rangle$
- For now, we will assume the states and actions are the same as in the real problem
- The model **approximates** the state transitions and rewards $\hat{p}_\eta\approx p$  
  $$  
  R_{t+1}, S_{t+1} \sim \hat{p}_\eta (r, s'|S_t,A_t)
  $$  
  (Note there is not probability distribution function)

- Optionally, we could model rewards and state dynamics separately

---
### 3. Model Learning
- Goal: **estimate** model $\mathcal{M}_\eta$ from experienve ${S_1, A_1, R_2,\ldots,S_T}$
- This is a supervised learning problem  
  $$
  \begin{aligned}
  S_1,A_1 &\rightarrow R_2, S_2 \\
  &\vdots \\
  S_{T-1},A_{T-1} &\rightarrow R_{T}, S_{T}
  \end{aligned}
  $$
- Learn a function $f(s,a) = r,s'$
- Pick loss function (e.g. mean-squared error), and find parameters $\eta$ that minimise empirical loss
- This would give an expectation model
- If $f(s,a)=r,s'$, then we would hope $s'\approx \mathbb{E}[S_{t+1}|s=S_t,a=A_t]$

---
### 4. Stochastic Model

- We may not want to assume everything is linear
- Then expected states may not be roght --- they may not correspond to actual states, and iterating the model may do weird things.
- Alternative: **stochastic models** (also know as **generative models**) 
  $$
  \hat{R}_{t+1}, \hat{S}_{t+1} = \hat{p}(S_t, A_t,\omega)
  $$
  where $\omega$ is a noise term
- Stochastic models can be chained, even if the model is non-linear.
- But they do add not noise 

---
### 5. Exmples of Models
- Table Looking Model
- Linear Expectation Model
- Linear Gaussian Model
- Deep Neural Network Model
- ......

#### 5.1 Table Lookup Model
- Model is an explicit MDP
- Count visits $N(s,a)$ to each state action pair  
  $$
  \begin{aligned}
  \hat{p}_t(s'|s,a) &= \frac{1}{N(s,a)}\sum_{k=0}^{t-1}I(S_k=s,A_k=a,S_{k+1}=s') \\
  \mathbb{\hat{p}_t}[R_{t+1}|S_t=s,A_t=a] &= \frac{1}{N(s,a)}\sum_{k=0}^{t-1}I(S_k=s,A_k=a)R_{k+1}
  \end{aligned}
  $$
- Alternatively, use non-parameteric 'replay' 
  - At each time-step t, record experience tuple $\langle S_t,A_t,R_{t+1},S_{t+1} \rangle$
  - To sample model, randomly pick tuple matching $\langle s,a,\cdot,\cdot\rangle$

---
### 6. Planing with a Model
- Given a model $\hat{p}_\eta$
- Solve the MDP $\langle \mathcal{S,A},\hat{p}_\eta \rangle$
- Using favourite planning algorithm
  - Value iteration
  - Policy iteration
  - Tree search
  - ......

#### 6.1 Sample-Based Planning
- A simple but powerful approach to planning
- Use the model **only** to generate sampler
- **Sample** experience from model  
  $$
  S,R \sim \hat{p}_\eta(\cdot|s,a)
  $$
- Apply **model-free** RL to sample, e.g.:
  - Monte-Carlo control
  - Sarsa
  - Q-learning

---  
### 7. Conventional model-based and model-free metheds

Traditional RL algorithms did not explicitly store their experiences, and were often placed into one of two groups.
  - **Model-free** methods update the value function and/or policy and do not have explicit dynamics models.
  - **Models-based** methods update the transition and reward models, and compute a value function or policy from the model.

---
### 8. Using experience in the place of model
Recall prioritized sweeping from tabular dynamic programming
  - Update the value function of the states with the largest magnitude Bellman errors using a priority queue.
A related idea is prioritized experience replay (Schaul et al, 2015) which works from experience for general function approximation.
  - The experience replay buffer maintain a priority for each transition, with the priority given by the magnitude of the Bellman error.
  - Minibatches are sampled using this priority to quickly reduce errors.
  - Weighted importance sampling corrects for bias from non-uniform sampling

---
### 9. Limits of Planning with an Inaccurate Model
- Given an imperfect model $\hat{p}_\eta \neq p$
- Performance is limited to optimal policy for approximate MDP $\langle \mathcal{M,A,\hat{p}_\eta}\rangle$ 
- Model-based RL is only as good as the estimated model
- When the model is inaccurate, planning process will compute a suboptimal policy (not covered in these slides)
  - Approach 1: when model is wrong, use model-free RL
  - Approach 2: reson explicitly about model uncertainty over $\eta$ (e.g. Bayesian methon)
  - Approach 3: Combine model-based and model-free methods in a safe way.

---
### 10. Real and Simulated Experience
We consider two sources of experience
Real experience Sampled from environment (true MDP)  
$$
r,s'\sim p
$$  
Sumulated experience Sampled from model (approximate MDP)  
$$
r,s' \sim \hat{p}_\eta
$$

---
### 11. Intergrating Learning and Planning 
- Model-Free RL
  - No model 
  - **Learn** value function (and/or policy) from real experience
- Model-Based RL (using Sample-Based Planning)
  - Learn a model from real experience 
  - **Plan** value function (and/or policy) from simulated experience
- Dyna
  - Learn a model from real experience
  - **Learn AND plan** value function (and/or policy) from real and simulated experience 
  - Treat real and sumulated esperience equivalently. Conceptually, the update from learning or planning are not distinguished.

#### 11.1 Dyna-Q Algorithm

>$$
\begin{aligned}
& \text{Initialize}\space Q(s,a)\space \text{and Model}(s,a)\quad \text{for all}\space s\in\mathcal{A}(s) \\
& \text{Do forever:} \\
& \qquad (a) \quad s \leftarrow \text{current (nonterminal) state} \\
& \qquad (b)\quad a \leftarrow \epsilon\text{-greedy}(s,Q) \\
& \qquad (c) \quad \text{Execute action}\space a\text{; observe resultant state, }s'\text{, and reward, }r\\
& \qquad (d)\quad Q(s,a)\leftarrow Q(s,a)+\alpha[r + \gamma\max_{a'}Q(s',a')-Q(s,a)]\\
& \qquad (e)\quad Model(s,a) \leftarrow s',r \quad\text{(assuming deterministic environment)}\\
& \qquad (f)\quad \text{Repeat}\space N \space \text{times:} \\
& \qquad\qquad s \leftarrow \text{random previously observed state} \\
& \qquad\qquad a\leftarrow \text{random action previously taken in}\space s \\
& \qquad\qquad s',r \leftarrow Model(s,a) \\
& \qquad\qquad Q(s,a)\leftarrow Q(s,a) + \alpha[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]
\end{aligned} 
>$$

### 11.2 Dyna with Function Approximation
- How can an agent plan when the actual environmental states are not know?
- Can directly approximate probability distributions of the transitions and the rewards.
- Probability distribution models in high dimensional feature spaces are computationally expensive and often inaccurate!

---
### 12. Planning for Action Selection
- We considered the case where planning is used to improve a global value function
- Now consider planning for the near future, to select the next action
- The distribution of states that may be encounted from **now** can diff from the distribution of states encountered from a starting state
- The agent may be able to make a more accurate local value function (for the states that will be encountered soon) than the global value function

#### 12.1 Forward Search
- **Forward search** algorithms select the best action by **look ahead**
- They build a search tree with the current state $s_t$ at the root
- Using a **model** of the MDP to look ahead
  ![tree](https://i.loli.net/2019/12/24/QkTADdw2EPx4zyG.png)
  
- No need to solve whole MDP, just sub-MDP starting from **now**

#### 12.2 Simulation-Based Search
- **Forward** search paradigm using sample-based planning
- **Simulate** episodes of experience from **now** with the model
- Apply **model-free** RL to simulated episodes  
  ![Search Path](https://i.loli.net/2019/12/24/JmliqxjnY7TB5fI.png)
    
- Simulate episode of experience from **now** with the model  
  $$
  \{S_t^k, A_t^k,R_{t+1}^k,\ldots,S_T^k\}_{k=1}^K \sim \hat{p}_\eta
  $$
- Apply **model-free** RL to sumulated episodes
  - Monte-carlo control $\rightarrow$ Monte-Carlo search
  - Sarsa $\rightarrow$ TD search

#### 12.3 Search tree vs. value function approximation
- Search tree is a table lookup approach 
- Based on a partial instantiation of the table 
- For model-free reinforcement learning, table lookup is naive 
  - Can't store value for all states
  - Doesn't generalise between similar states
- For simulation-based search, table lookup is less naive
  - Search tree stores value for easily reachable states
  - In huge search spaces, value function approximation is helpful

### 13. Monte-Carlo Simulation
- Given a parameterized model $\mathcal{M}_\eta$ and a **simulation policy** $\pi$
- Simulate $K$ episodes from current state $S_t$  
  $$
  \{S_t^k=S_t,A_t^k,R_{t+1}^k,S_{t+1}^k,\ldots,S_t^k\}_{k=1}^K \sim \hat{p}_\eta, \pi
  $$
- Evaluate state by mean return (**Monte-Carlo evaluaiton**)  
  $$
  v(\color{red}{S_t})=\frac{1}{K}\sum_{k=1}^K G_t^k \rightsquigarrow v_\pi(S_t)
  $$

#### 13.1 Simple Monte-Carlo Search
- Given a model $\mathcal{M}_\eta$ and a policy $\pi$
- For each action $a \in \mathcal{A}$
  - Simulate $K$ episodes from current (real) state $s$  
  $$
  \{S_t^k=s,A_t^k=a,R_{t+1}^k,S_{t+1}^k,A_{t+1}^k,\ldots,S_t^k\}_{k=1}^K \sim \mathcal{M}_v,\pi
  $$
  - Evaluate actions by mean return (Monto-Carlo evaluation)  
  $$
  q(\color{red}{s,a})=\frac{1}{K}\sum_{k=1}^K G_t^k \rightsquigarrow q_\pi(S_t)
  $$
  - Select current (real) action with maximum value  
  $$
  A_t=\mathop{\text{argmax}}\limits_{a\in\mathcal{A}}q(S_t,a)
  $$

#### 13.2 Monte-Carlo Tree Search (Evaluation)
- Given a model $\mathcal{M}_\eta$
- Simulate $K$ episodes from current state $S_t$ using current simulation policy $\pi$  
  $$
   \{\color{red}{S_t^k=S_t},A_t^k,R_{t+1}^k,S_{T+1}^k,\ldots,S_t^k\}_{k=1}^K \sim \mathcal{M}_v,\pi
  $$
- Build a search tree containing visited states and actions
- **Evaluate** states $q(s,a)$ by mean return of eqisodes from $s,a$  
  $$
  q(s,a)=\frac{1}{N(s,a)}\sum_{k=1}^K\sum_{u=t}^T1(S_u^k,A_u^k=s,a)(G_u^k \rightsquigarrow q_\pi(s,a))
  $$  
- After searching, select current (real) action with maximum value in search tree  
  $$
  a_t=\mathop{\text{argmax}}\limits_{a\in\mathcal{A}}q(S_t,a)
  $$

#### Monte-Carlo Tree Search (Simulation)
- In MCTS, the simulation policy $\pi$ **improves**
- The simulation policy $\pi$ has two phases (in-tree, out-of-tree)
  - **Tree policy** (improves): pick action from $q(s,a)$ (e.g. $\epsilon$-greedy($q(s,a)$))
  - **Rollout policy** (fixed): e.g., pick actions randomly
- Repeat (for each simulated episode)
  - **Select** actions in tree according to tree policy
  - **Expand** search tree by one node
  - **Rollout** to termination with default policy
  - **Update** action-values $q(s,a)$ in the tree
- Output best action when simulation time runs out.
- With some asumptions, converges to the optimal values, $q(s,a)\Rightarrow q_*(s,a)$
