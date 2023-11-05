# **MAMOC1**

## Abstract

 This study focuses on methods of computationally modelling systems of many interacting particles. One is a class of methods called reaction-diffusion systems, which are mathematical constructs that describe how spatial and temporal patterns emerge from the interactions between organisms and their environment. Also considered are agent based models, which instead only use hyper-local information and do not rely on macroscopic differential equations at all. In particular, the predator prey system is considered from both perspectives. By coupling local interaction rules with movement, these models elucidate the dynamics of population spread, predator-prey relationships, and the emergence of patterns in ecosystems.

## Introduction

Theoretical ecology is a mathematical framework for understanding how the interactions of individual organisms with each other and with the environment determine the distribution of populations and the structure of communities. Many different models are needed, each based on some set of hypotheses about the scale and structure of the spatial environment and the way that organisms disperse through it. When many agents interact in simple ways, sometimes emergent properties can be observed in their collective, such as the propagation of wavefronts or the formation of patterns. 
These patterns are glimpses of structure from chaos, and have prompted countless minds to become fascinated by the connections between maths and biology.

## Mathematics of Reaction-Diffusion Systems

Reaction-diffusion models are a way of translating the simple assumptions about ways agents can move and interact on the local level into global conclusions about the persistence or extinction of populations and the coexistence of interacting species. Born from the macroscopic observations of molecular diffusion, reaction-diffusion models now more widely refer to any event-driven system of interacting moving agents. In biology, the great success of reaction-diffusion is the Turing model, which describes how homogeneous groups of cells in an embryo can spontaneously differentiate into patterns, like the spots and stripes on animal skins. In chemistry, the Belousov-Zhabotinsky chemical reaction displays incredible dispersive concentric patterns. Further examples are as distributed as econometric information diffusion, as  and as grievous as modelling the spread of forest fires. 

The general form of reaction-diffusion differential equation systems is given by 

$$\partial_{t} q_{i} =  D_{m} \nabla^{2} q_{i} + R_{j}(q_{j})$$

where $q_{i}$ describes a concentration, $D_{m}$ the diffusion coefficient, and $R_{j}(q_{j})$ a function of concentration representing agents local behaviour - i.e. the birth and death rate of the agents. The first term on the right hand side can be recognised as Fick’s law.

The dynamics of our model is specified by the rates at which individuals move and die or reproduce. As such, a good place to start is to consider the local mechanics of our agent movement and reactions, supposing that agents may only move to a randomly chosen nearest neighbour of their location, and reproduce or die at rates which depend on the number of individuals at the same location. Using some simplifying assumptions along the way, there are creative ways to derive the reaction term, which is usually, hopefully, linear in $q_i$. Often, however, we are not so lucky. Spatial models often unavoidably invite non-linear terms, resulting in chaotic long-term behaviour.

## Activator-inhibitor systems

Activator-inhibitor systems involve two interacting species where one of them, the activator, enhances the production of both species, whereas the inhibitor works to slow down the production of both. Beyond activator-inhibitor's usual biochemical applications, ecology uses it in the form of the predator prey system, and in particular the Lotka-Volterra equations. 

### Lotka-Volterra equations.

The Lotka-Volterra equations model the population dynamics of a predator species and a prey species, perhaps foxes and rabbits, and is a convinient way to explore how reaction terms work for basic diffusion-reactive systems. Since we want to completely ignore spatial dimensions, we can use the mean field assumption: that all agents interact with the average effect of all others. Letting $x_1(t)$ denote the prey population and $x_2(t)$ denote the predator population at time $t$, we get

$$\frac{dx_1}{dt} = \alpha x_1 - \beta x_1 x_2,
\frac{dx_2}{dt} = \delta x_1 x_2 - \gamma x_2.$$

Here:

- $\alpha$ is the natural birth rate of the prey in the absence of predators.

- $\beta$ is the death rate of the prey due to predation.

- $\gamma$ is the natural death rate of the predators in the absence of prey.

- $\delta$ is the rate at which predators increase by consuming prey.

The steady states of this system of differential equations are found by setting the time derivatives to zero. The trivial solution $$x_1 = x_2 = 0$$ indicates mutual extinction, but otherwise, solving for constants, we get ${x_1, x_2} = {\gamma / \delta, \alpha / \beta}$. We would like to analyse the stability of these steady state solutions. To do this, we linearise our set of differential equations, so that we have

$$J = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} \end{pmatrix} = \begin{pmatrix} \alpha - \beta x_2 & -\beta x_1 \\
\delta x_2 & \delta x_1 - \gamma \end{pmatrix}  = \begin{pmatrix} 0 & -\beta \frac{\gamma}{\delta} \\
\delta \frac{\alpha}{\beta} & 0 \end{pmatrix}.$$

The trivial solution has eigenvalues $\alpha$ and $-\gamma$, indicating a saddle point. This is good, because saddle points are unstable, meaning the model does not predict spiralling uncontrollably towards extinction. The other matrix has eigenvalues $+i \sqrt{\alpha \gamma}$ and $- i \sqrt{\alpha \gamma}$, which indicates periodic trigonometric-esque solutions.

//// Insert photo of population in typical lotka volterra

The model elegantly captures the core ideas: prey populations grow naturally but are eaten by predators, and predator populations decline without food but grow when they eat prey.

Adding in our Fick’s law inspired diffusion terms, we get the spatial Lotka-Volterra equations 

$$\frac{\partial x_1}{\partial t} = \alpha x_1 - \beta x_1 x_2 + D_1 \nabla^2 x_1 $$

$$\frac{\partial x_2}{\partial t} = \delta x_1 x_2 - \gamma x_2 + D_2 \nabla^2 x_2 $$

This set of differential equations can be modelled computationally on a randomised initial population to show the progression of the reaction-diffusion system. We can use an Euler-esque method for executing this progression, by multiplying our differential equations by a timestep $dt$ to get a $dX$, and adding that to $X$ each iterative timestep $t$. 

```python
steps = 2000
for t in range(steps):

    # Lotka-Volterras with diffusion laplacians
    laplacian_X1 = lapalcian(X1)
    laplacian_X2 = lapalcian(X2)
    dX1 = (alpha * X1 - beta * X1 * X2 + Dx * laplacian_X1) * dt
    dX2 = (-gamma * X2 + delta * X1 * X2 + Dy * laplacian_X2) * dt

    X1 += dX1
    X2 += dX2
```

Animated, we get our first look at the emergent patterns of spatial ecology.

// insert animation

### Computational agent based models for two species population dynamics.

Spatial ecology can also be modelled computationally using agent-based models, and the benefit of this is that it did not assume the truth of a set of differential equations. As such, agent based models offer a completely different method for simulating systems of interacting particles by operating exclusively at the local level, with no macroscopic information shared to individual agents. Diffusion occurs via steps in random directions. Reactions occur based on only local information, such as if two interacting agents of the same species at the same place have enough energy to procreate. While these models are much closer approximations to real life, they still lack the macroscopic dynamics of flocking, herding, fleeing, or hunting. As such, it may make sense to imagine these agents as small groups of microorganisms, diffusing due to the random flow of Brownian motion, rather than intelligent autonomous animals.

The iterative time steps of my program is as simple as 

```python
TheGrid = ReactionDiffusionLibrary.Grid(gridxsize, gridysize)
Prey = ReactionDiffusionLibrary.Species("Prey", Prey_E0, Prey_EP, Prey_N, TheGrid)
Pred = ReactionDiffusionLibrary.Species("Predator", Pred_E0, Pred_EP, Pred_N, TheGrid)

steps = 2000
for step in range(steps):
    Prey.Move()
    Pred.Move()
    TheGrid.Interact(Prey, Pred)
```
where the Prey, Pred, and TheGrid instances are C# classes incorporated into the python script using pythonnet clr. The two main mechanisms of reaction-diffusion are contained in the ``Move`` method and the ``Interact`` method. Here is ``Move``,

```c#
public void Move()
    {
        if (AgentsList.Count < 1000) Dying = true;
        else Dying = false;

        List<double[]> SpeciesCoords = new List<double[]>(AgentsList.Count);
        Random random = new Random();

        for (int i = 0; i < AgentsList.Count; i++)
        {
            var agent = AgentsList[i];
            int dirIndex = random.Next(0, 5);
            agent.Move(Grid.Directions[dirIndex]);
            SpeciesCoords.Add(new double[] {agent.X, agent.Y});
            
            char yOrDCondition = PredOrPrey == "Prey" ? 'y' : 'd';
            agent.AgentId = $"{yOrDCondition}{i}";
            agent.AgentIndex = i; 

            Grid.AgentsInGrid[(int)agent.X][(int)agent.Y].Add(agent.AgentId);
        }
        this.Grid.WriteAgentsInGridToFile();
    }
```

Each iteration, the coordinates of all the agents in the species are rewritten based on their movement in a random direction, and this also helped with dealing with the non constant number of agents due to births and deaths.

The one alteration I made to the program to make it not a strictly agent based model is the Dying boolean, which turns on when a species’’ total number of agents dips below 1000 gridwide. This allowed agents to procreate on their own whenever the species has sufficiently low numbers rather than needing to randomly stumble upon another of their species on the grid. To justify this, we can imagine each dot as a small collection of microorganisms rather than an individual. 

The Interact() method iterates through each square in the grid, collecting pairs of agents. If two agents of the same species are in the same grid square, and they have above the minimum procreation energy, they will breed and produce another agent at that square. If a predator is in the same square as a prey, it will feed on the prey, killing it and absorbing its energy. 

```c#
    public void Interact(Species PreySpeciesObj, Species PredSpeciesObj)
    {
        int ySize = GridYSize;
        int xSize = GridXSize;
        for (int y_i = 0; y_i < ySize; y_i++)
        {
            for (int x_i = 0; x_i < xSize; x_i++)
            {
                List<string> agents = AgentsInGrid[y_i][x_i];
                if (agents.Count == 0) continue;
                
                List<int> preys = new List<int>();
                List<int> preds = new List<int>();

                // Check each agent in a cell
                foreach (string agentStr in agents)
                {
                    if (agentStr.StartsWith('y')) preys.Add(int.Parse(agentStr.Substring(1)));
                    if (agentStr.StartsWith('d')) preds.Add(int.Parse(agentStr.Substring(1)));
                }

                // Predators breed and feed on prey
                Procreate(preds, PredSpeciesObj);
                for (int i = 0; i < preds.Count; i++)
                {
                    if (i < preys.Count)
                    {
                        Agent predator = PredSpeciesObj.AgentsList[preds[i]];
                        Agent food = PreySpeciesObj.AgentsList[preys[i]];
                        predator.Energy += food.Energy;
                        food.AddToDeathList();
                    }
                }

                // Preys breed
                Procreate(preys, PreySpeciesObj);

                // Check for special procreation cases
                if (PreySpeciesObj.Dying && preys.Count > 0) SingleProcreation(preys[0], PreySpeciesObj);
                if (PredSpeciesObj.Dying && preds.Count > 0) SingleProcreation(preds[0], PredSpeciesObj);
            }
        }

        // At the end of the turn, make babies into adults and handle deaths
        PreySpeciesObj.NewDay();
        PredSpeciesObj.NewDay();
        ClearAgentsInGrid();
    }
```

The results are different from the Lotka-Volterra animation. 

// Animation of reaction diffusion.

Instead of smoothly fading between areas of high prey density and high predator density, we see vast areas of blank space. But still, some similar emergent properties can be seen, in particular the collections of agents evolving in wavefronts.


## Deriving the Turing instability of the Lotka-Volterra equations

To prove that the given system of equations produces Turing patterns, we need to perform a linear stability analysis around a homogeneous steady state and look for conditions under which perturbations grow in a spatially inhomogeneous way.

First, let's find the homogeneous steady state $( x_1^{\*}, x_2^{\*} )$ for a spatially uniform state by setting 
$$\frac{\partial x_1}{\partial t} = \frac{\partial x_2}{\partial t} = 0, \nabla^2 x_1 = \nabla^2 x_2 = 0$$

Subbing into eqn [?] gives
$$0 = \alpha x_1^{\*} - \beta x_1^* x_2^{\*}, \\
0 = \delta x_1^{\*} x_2^{\*} - \gamma x_2^{\*}.$$

Solving for $x_1^{\*}$ and $x_2^{\*}$,
$$x_2^* = \frac{\alpha}{\beta}$$ 
$$x_1^* = \frac{\gamma}{\delta x_2^*} = \frac{\gamma \beta}{\alpha \delta}$$

Next, we linearize the system around the steady state by letting $x_1 = x_1^{\*} + u$ and $x_2 = x_2^{\*} + v$, where $u$ and $v$ are small perturbations. We then substitute these into our system and keep only terms that are linear in $u$ and $v$.

After linearization, the system becomes:
$$\partial_t u = \alpha u - \beta (x_1^{\*} v + x_2^{\*} u) + D_1 \nabla^2 u,$$
$$\partial_t v = \delta (x_1^{\*} v + x_2^{\*} u) - \gamma v + D_2 \nabla^2 v$$

Substituting the steady-state values:
$$\partial_t u = (\alpha - \beta x_2^{\*}) u - \beta x_1^{\*} v + D_1 \nabla^2 u,$$
$$\partial_t v = \delta x_2^{\*} u + (\delta x_1^{\*} - \gamma) v + D_2 \nabla^2 v$$

So, we have our fully linearized system which we can write in matrix form as:

$$\begin{pmatrix} \partial u \\ 
\partial v \end{pmatrix} = \begin{pmatrix} \alpha - \beta x_2^* + D_1 \nabla^2 & -\beta x_1^* \\ 
\delta x_2^* & \delta x_1^* - \gamma + D_2 \nabla^2 \end{pmatrix} \begin{pmatrix} u \\ 
v
\end{pmatrix}$$


Now we look for solutions in the form of $e^{\lambda t + ik\cdot r}$, where $\lambda$ is the growth rate of the perturbation, $k$ is the wave number, and $r$ is the position vector.

We substitute this form into the linearized system and solve for $\lambda$. The dispersion relation is obtained by setting the determinant of the matrix minus $\lambda I$ to zero:

$$\det
\begin{pmatrix} D_1 k^2 - \lambda & -\frac{\beta \gamma}{\delta} \\
\frac{\alpha \delta}{\beta} & D_2 k^2 - \gamma - \lambda \end{pmatrix}= 0$$

Solving the characteristic equation, we get:
$$(D_1 k^2 - \lambda)(D_2 k^2 - \gamma - \lambda) - \left(-\frac{\beta \gamma}{\delta}\right)\left(\frac{\alpha \delta}{\beta}\right) = 0$$

Expanding and simplifying gives us a quadratic equation in $\lambda$,
$$\lambda^2 - \lambda(D_1 k^2 + D_2 k^2 - \gamma) + D_1 D_2 k^4 - \gamma D_1 k^2 - \alpha \gamma = 0,$$
and subbing in for values I used for Figure [1],
$$\alpha = 0.1, \quad \beta = 0.02, \quad \gamma = 0.3, \quad \delta = 0.01, \quad D_1 = 0.1, \quad D_2 = 0.05,$$
we get to our final characteristic equation
$$\lambda^2 - 0.15 \lambda k^2 + 0.3 \lambda + 0.005 k^4 - 0.03 k^2 - 0.15 = 0.$$


For Turing instability, we need:
1. the diffusion terms must destabilize the homogeneous steady state, meaning the real part of $\lambda$ must be positive for some range of $k$,
2. and the homogeneous system must be stable without diffusion, implying that $\lambda < 0$ when $k = 0$

And so, we look for any sign change in the real part of $\lambda (k)$, which indicates the presence of a critical wave number $k_c$ beyond which the system becomes unstable to perturbations and patterns will emerge. If such a sign change exists within the physically meaningful range of $k$ (usually $k > 0$), we can confirm the presence of Turing instability.




