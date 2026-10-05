"""Script version of Fall_26_CP1_Advanced_Starter_Notebook.ipynb.

Run from the repo root (through Slurm: sbatch -J advanced slurm/run.sbatch advanced_starter.py).
Figures are written to outputs/advanced/ instead of being shown inline.
"""

# %% cell 1
import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Disable GPU for JAX (Remove if you want to use GPU)
os.environ.setdefault("TQDM_MININTERVAL", "30")  # keep progress bars from flooding the job log

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display on compute nodes
import matplotlib.pyplot as plt

FIG_DIR = "outputs/advanced"
os.makedirs(FIG_DIR, exist_ok=True)

def save_figs(cell):
    # stand-in for inline notebook display: save every open figure, then close them
    for k, num in enumerate(plt.get_fignums()):
        plt.figure(num).savefig(f"{FIG_DIR}/cell{cell:02d}_{k}.png", dpi=120, bbox_inches="tight")
    plt.close("all")
import random
from tqdm.auto import tqdm, trange

# deteministic random numbers
np.random.seed(0)
random.seed(0)
save_figs(1)

# %% [markdown] cell 2
# # Linkage Synthesis Challenge Problem Advanced Code
# Here we will tackle the problem with a more complex GA approach. In this notebook, rather than just optimizing the node positions, we will also optimize the connectivity of the mechanims itself. First let's just load the kangaroo target curves like we did before:

# %% cell 3
target_curves = np.load('kangaroo_target_curves.npy')

# Names of the three targets, ordered easiest -> hardest (same as the starter notebook)
TARGET_NAMES = [
    'Kangaroo 1 (round body)',
    'Kangaroo 2 (no ears, no tail)',
    'Kangaroo 3 (full meme)',
]

# Each target has its own reference point / constraint bounds. Imported from LINKS.CP
# so these are exactly the values evaluate_submission() scores you against.
from LINKS.CP import REFERENCE_POINTS

# >>> Switch which kangaroo you are working on here (0, 1 or 2). <<<
# Everything downstream (target curve, reference point, constraints) follows this.
target_index = 1

target_curve = np.array(target_curves[target_index])
reference_point = REFERENCE_POINTS[target_index]

print(f"Working on Problem {target_index + 1}: {TARGET_NAMES[target_index]}")
print(f"Reference point -> distance <= {reference_point[0]}, material <= {reference_point[1]}")

# Plot all target curves

# Initialize a subplot for plotting all target curves
fig, axs = plt.subplots(1, target_curves.shape[0], figsize=(8, 5))

# Loop through the target curves to plot them
for i in range(target_curves.shape[0]):
    # Extract x and y coordinates from the target curve
    x_coords = np.array(target_curves[i])[:, 0]
    y_coords = np.array(target_curves[i])[:, 1]

    # Plot the curve on the respective subplot
    axs[i].plot(x_coords, y_coords, color='black', linewidth=3)

    # Set title for each subplot
    axs[i].set_title(TARGET_NAMES[i].replace(' (', '\n('))

    # Ensure equal scaling for x and y axes
    axs[i].axis('equal')
    axs[i].axis('off')

plt.tight_layout()
save_figs(3)

# %% [markdown] cell 4
# ## GA Full Mechanism Generation
# To set up the problem within Pymoo we will have to come up with a 1D representation of the mechanisms. This code effectively just flattens the matrices and variables describing our mechanism into a single vector. However, this representation scheme is highly suboptimal, and can be improved significantly. As a demonstration, we will slightly improve upon this representation, though we encourage you to take this further. As we discussed, we can replace the edges with a connectivity matrrix. We also know the connectivity matrix is symmetric (if nodes 1 and 2 are connected nodes 2 and 1 are also connected) and has a diagonal equal to zero. To avoid making the optimization come up with the whole connectivity matrix, we will only have it come up with the upper triangular half of the matrix (not including the diagonal). This means the algorithm has to come up with $\frac{N^2-N}{2}$ 0s or 1s (i.e., booleans) to construct the adjacency matrix. We can further simplify the process by selecting a fixed motor position and having the optimization build the rest of the mechanism, removing the need for the optimization to determine the motor as well as reducing the number of 0s and 1s in the upper triangular portion by one (Saving 3 parameters in the representation). To do this we will start with the following mechanism:
#
# <img src="https://i.ibb.co/mGywWgr/start.jpg" alt="start" border="0">
#
# We set Node 0 to ground and Node 1 to be ordinary and the motor to be `[1,0]`. Now we know that the upper triangular portion of the connectivity matrix starts with:
#
# <img src="https://i.ibb.co/5RqKYQN/upper-triangular-fromstart.jpg" alt="upper-triangular-fromstart" border="0">
#
# Since we know the types of the first two nodes, we save another two parameters from the optimization. At this point we will have the same representation as we have before, but with fewer numbers and no motor or N (size of mechanisms) as we generate the mechanism to have a certain size.
#
# <img src="https://i.ibb.co/gv2D0Y5/Optimization-Represenation.png" alt="Optimization-Represenation" border="0">
#
# Now that we have a (slightly) improved 1D representation scheme, we will set up our problem with this many parameters. We also need to specify our constraints! Inequality constraints can be set in Pymoo (See: [constraints in pymoo](https://www.pymoo.org/constraints/index.html)) as less than or equal to zero constraints. We will set the two constraints on distance and material use in our problem.
#
# So based on this lets get started with defining the problem in Pymoo.
# <ul>
#     <li>
#         <code>__init__</code>: We need to differentiate between datatypes because crossover, sampling, and mutation will happen slightly differently based on datatype. As such, we create a dictionary of variables and specify which type of variable each one is.
#     </li>
#     <li>
# <code>convert_1D_to_mech</code>: Reformats our dictionary of variables into design info that our simulator uses (edges, x0, etc.). You will need to update this function if you change the way you parameterize mechanisms. IMPORTANT: This function should not be confused for the. We have also provided a function, <code> convert_mech_to_1D </code> to do the opposite, which you may find useful. It is currently unused and is commented out.
#     </li>
#     <li>
# <code>_evaluate</code>: Next, we define the evaluation function. We construct the mechanism using <code>convert_1D_to_mech</code>. Next, perform the same evaluations as we did in the starter notebook.
#     </li>
# </ul>

# %% cell 5
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling, Sampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize

from LINKS.Optimization import DifferentiableTools, Tools
save_figs(5)

# %% cell 6
PROBLEM_TOOLS = Tools( # we have to define this outside the class due to pymoo deepcopy limitations
            device='cpu' # device to run the optimization on
        )
PROBLEM_TOOLS.compile() # compile the functions for faster runs

class mechanism_synthesis_optimization(ElementwiseProblem):

    # When intializing, set the mechanism size and target curve
    def __init__(self, target_curve, N = 5, reference_point=None):
        self.N = N
        variables = dict()

        # The upper triangular portion of our NxN Connectivity Matrix consists of Nx(N-1)/2 boolean variables:
        for i in range(N):
            for j in range(i):
                variables["C" + str(j) + "_" + str(i)] = Binary()

        # We Delete C0_1 since we know node 1 is connected to the motor
        del variables["C0_1"]

        #Our position matrix consists of Nx2 real numbers (cartesian coordinate values) between 0 and 1
        for i in range(2*N):
            variables["X0" + str(i)] = Real(bounds=(0.0, 1.0))

        # Our node type vector consists of N boolean variables (fixed vs non-fixed)
        for i in range(N):
            variables["fixed_nodes" + str(i)] =  Binary(N)

        # Our target node is an integer between 1 and N-1, (any except the motor node).
        variables["target"] = Integer(bounds=(1,N-1))

        # Set up some variables in the problem class we inherit for pymoo
        # n_obj=number of objectives, n_constr=number of constraints
        # Our objectives are chamfer distance and material, and they both have constraints.
        super().__init__(vars=variables, n_obj=2, n_constr=2)

        # Store the target curve point cloud
        self.target_curve = target_curve

        # Store this target's reference point, which doubles as the constraint
        # upper bounds (distance, material) for this curve
        if reference_point is None:
            reference_point = REFERENCE_POINTS[target_index]
        self.reference_point = np.array(reference_point)


    def convert_1D_to_mech(self, x):
        N = self.N

        # Get target joints index
        target_idx = x["target"]

        # Build connectivity matrix from its flattened constitutive variables
        C = np.zeros((N,N))
        x["C0_1"] = 1

        for i in range(N):
            for j in range(i):
                # C[i,j] = x["C" + str(j) + "_" + str(i)]
                C[j,i] = x["C" + str(j) + "_" + str(i)]

        edges = np.array(np.where(C==1)).T

        # Reshape flattened position matrix to its proper Nx2 shape
        x0 = np.array([x["X0" + str(i)] for i in range(2*N)]).reshape([N,2])

        # Extract a list of Nodes that are fixed from boolean fixed_nodes vector
        fixed_joints = np.where(np.array([x["fixed_nodes" + str(i)] for i in range(N)]))[0].astype(int)

        #We fix the motor and original ground node as 0 and 1 respectively in this implementation
        motor=np.array([0,1])

        return x0, edges, fixed_joints, motor, target_idx

    def convert_mech_to_1D(self, x0, edges, fixed_joints, target_idx=None, **kwargs):
        # This function assumes motor to be [0, 1] our random mechanism generator automatically does this
        N = self.N

        # Initialize dictionary to store 1D representation of mechanism
        x = {}

        # Store target node value
        if target_idx is None:
            target_idx = x0.shape[0]-1 # Assume last node is the target if not specified

        x["target"] = target_idx

        # Store connectivity matrix in its flattened form
        C = np.zeros((N,N), dtype=bool)
        C[edges[:,0], edges[:,1]] = 1
        C[edges[:,1], edges[:,0]] = 1

        for i in range(N):
            for j in range(i):
                x["C" + str(j) + "_" + str(i)] = C[i,j]

        del x["C0_1"]

        # Store position matrix in its flattened form
        if x0.shape[0] != N:
            x0 = np.pad(x0, ((0, N - x0.shape[0]), (0, 0)), 'constant', constant_values=0)

        for i in range(2*N):
            x["X0" + str(i)] = x0.flatten()[i]

        # Store fixed nodes in boolean vector form
        for i in range(N):
            x["fixed_nodes" + str(i)] = (i in fixed_joints) or (i>=N)

        return x

    def _evaluate(self, x, out, *args, **kwargs):
        #Convert to mechanism representation
        x0, edges, fixed_joints, motor, target_idx = self.convert_1D_to_mech(x)

        # Simulate
        distance, material = PROBLEM_TOOLS(x0,
                                edges,
                                fixed_joints,
                                motor,
                                self.target_curve,
                                target_idx=target_idx
                            )

        out["F"] = np.array([distance, material])
        # Constraints: distance <= reference_point[0], material <= reference_point[1]
        out["G"] = out["F"] - self.reference_point
save_figs(6)

# %% [markdown] cell 7
# Now let's try this with NSGA-II on the curve selected by `target_index` above. We will initialize with mechanisms with 6 nodes. You can test what happens if you increase this number. You should quickly discover that this naive parameterization does not perform well when initializing the GA with more complex mechanisms:

# %% cell 8
problem = mechanism_synthesis_optimization(target_curve, N=6, reference_point=reference_point)

algorithm = NSGA2(pop_size=100,
                  sampling=MixedVariableSampling(),
                  mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                #   mutation=PolynomialMutation(prob=0.9),
                  eliminate_duplicates=MixedVariableDuplicateElimination())

results = minimize(problem,
                algorithm,
                ('n_gen', 100),
                verbose=True,
                save_history=True,
                seed=0
                )
save_figs(8)

# %% cell 9
if results.X is None:
    print("No feasible solution found !!!!!")
save_figs(9)

# %% [markdown] cell 10
# You can see that if you just run GA you will not be able to find any good solutions, even for smaller mechanisms with 6 joints!
#
# In the starter notebook we gave you a mechanism to initialize GA with. Now we introduce the `MechanismRandomizer` class which can help you generate random mechanisms of any size you need. Let's see how this class works (The first time you run this it will be a little slow):

# %% cell 11
from LINKS.Optimization import MechanismRandomizer
from LINKS.Visualization import MechanismVisualizer
save_figs(11)

# %% cell 12
randomizer = MechanismRandomizer(
    min_size = 6, # smalllest mechanism to sample
    max_size = 14, # largest mechanism to sample
    device='cpu')

visualizer = MechanismVisualizer()
save_figs(12)

# %% cell 13
random_mechanism = randomizer()

# Let's see what the output looks like (it is exactly like the submission format)
print(random_mechanism)
save_figs(13)

# %% cell 14
# Visualize the random mechanism
visualizer(**random_mechanism)
save_figs(14)

# %% cell 15
# Now let's generate 100 mechanisms of size 7 and initilize a population for optimization
mechanisms = [randomizer(n=7) for _ in trange(100)]
save_figs(15)

# %% cell 16
problem = mechanism_synthesis_optimization(target_curve, N=7, reference_point=reference_point)

initial_population = [problem.convert_mech_to_1D(**mech) for mech in mechanisms]

class sample_from_random(Sampling):
        def _do(self, problem, n_samples, **kwargs):
                return np.array([initial_population[i%len(initial_population)] for i in range(n_samples)])

F = problem.evaluate(np.array(initial_population))[0]
print(f'Best Distance Performance In random population: {F[:,0].min()}')
print(f'Best Material Performance In random population: {F[:,1].min()}')
save_figs(16)

# %% cell 17
algorithm = NSGA2(pop_size=100,
                  sampling=sample_from_random(),
                  mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                  mutation=PolynomialMutation(prob=0.5),
                  eliminate_duplicates=MixedVariableDuplicateElimination())

results = minimize(problem,
                algorithm,
                ('n_gen', 100),
                verbose=True,
                save_history=True,
                seed=123
                )
save_figs(17)

# %% [markdown] cell 18
# We see that if we initialize GA with some valid samples, it is much more likely to succeed! Let's take a look at the solutions of this run:

# %% cell 19
from LINKS.Visualization import GAVisualizer
from LINKS.Kinematics import MechanismSolver
from LINKS.Geometry import CurveEngine
save_figs(19)

# %% cell 20
from pymoo.indicators.hv import HV
ga_visualizer = GAVisualizer()
solver = MechanismSolver(device='cpu')
curve_engine = CurveEngine(device='cpu')

if not results.X is None:
    #Specify reference point (this target's own reference point)
    ref_point = np.array(reference_point)

    #Calculate Hypervolume
    ind = HV(ref_point)
    hypervolume = ind(results.F)

    #Print and plot
    print('Hyper Volume ~ %f' %(hypervolume))
    ga_visualizer.plot_HV(results.F, ref_point, objective_labels=['Distance', 'Material']) #use the plot_HV function from utils
else:
    print('Did Not Find Solutions!!')
save_figs(20)

# %% [markdown] cell 21
# Now Let's visualize the best solution for distance

# %% cell 22
if not results.X is None:
    if type(results.X)!=dict:
        best_idx = np.argmin(results.F[:,0])
        best_sol, edges, fixed_joints, motor, target_idx  = problem.convert_1D_to_mech(results.X[best_idx])
    else:
        best_sol, edges, fixed_joints, motor, target_idx  = problem.convert_1D_to_mech(results.X)
else:
    print('Did Not Find Solutions!!')
save_figs(22)

# %% cell 23
plt.figure(figsize=(8,8))
visualizer(best_sol, edges, fixed_joints, motor, highlight=target_idx, ax=plt.gca())

traced_curve = solver(best_sol, edges, fixed_joints, motor)[target_idx]
curve_engine.visualize_comparison(
    traced_curve,
    target_curve
)

print(f'Distance: {results.F[best_idx,0]:.4f}, Material: {results.F[best_idx,1]:.4f}')
save_figs(23)

# %% [markdown] cell 24
# Now Let's visualize the best solution for material

# %% cell 25
if not results.X is None:
    if type(results.X)!=dict:
        best_idx = np.argmin(results.F[:,1])
        best_sol, edges, fixed_joints, motor, target_idx  = problem.convert_1D_to_mech(results.X[best_idx])
    else:
        best_sol, edges, fixed_joints, motor, target_idx  = problem.convert_1D_to_mech(results.X)
else:
    print('Did Not Find Solutions!!')
save_figs(25)

# %% cell 26
plt.figure(figsize=(8,8))
visualizer(best_sol, edges, fixed_joints, motor, ax=plt.gca())

traced_curve = solver(best_sol, edges, fixed_joints, motor)[target_idx]
curve_engine.visualize_comparison(
    traced_curve,
    target_curve
)

print(f'Distance: {results.F[best_idx,0]:.4f}, Material: {results.F[best_idx,1]:.4f}')
save_figs(26)

# %% [markdown] cell 27
# You can use this population to make a submission:

# %% cell 28
from LINKS.CP import make_empty_submission, evaluate_submission

submission = make_empty_submission()

# add out population to submission
if not results.X is None:
    for i in range(results.X.shape[0]):
        if not isinstance(results.X, dict):
            x0_member, edges, fixed_joints, motor, target_idx  = problem.convert_1D_to_mech(results.X[i])
        else:
            x0_member, edges, fixed_joints, motor, target_idx  = problem.convert_1D_to_mech(results.X)

        mech = {
            'x0': x0_member,
            'edges': edges,
            'fixed_joints': fixed_joints,
            'motor': motor,
            'target_joint': target_idx
        }

        submission[f'Problem {target_index + 1}'].append(mech)
save_figs(28)

# %% cell 29
print(evaluate_submission(submission))
save_figs(29)

# %% [markdown] cell 30
# You can see that GA is still not doing great with handling different mechanisms. You could adjust the initialization more, change mutation and cross-over functionality. So far however, we have not yet touched on gradient-based optimization. What if we took the solutions of GA and optimized for each of the objectives.
#
# Now let's take the GA solutions from above and just optimize for the lowest distance. Then we will add the optimized solutions to the population we have and see how the hypervolume changes.
#
#  **Note that if the distance or material use exceeds this target's reference point during this optimization the score will not change How could you address this?**

# %% cell 31
from LINKS.Optimization import DifferentiableTools
save_figs(31)

# %% cell 32
differentiable_optimization_tools = DifferentiableTools(
    device='cpu' # device to run the optimization on
)
differentiable_optimization_tools.compile() # compile the functions for faster runs
save_figs(32)

# %% cell 33
# lets make a batch from our GA results
x0s = []
edges = []
fixed_joints = []
motors = []
target_idxs = []
if not results.X is None:
    if not isinstance(results.X, dict):
        for i in range(results.X.shape[0]):
            x0_member, edges_member, fixed_joints_member, motor_member, target_idx_member  = problem.convert_1D_to_mech(results.X[i])
            x0s.append(x0_member)
            edges.append(edges_member)
            fixed_joints.append(fixed_joints_member)
            motors.append(motor_member)
            target_idxs.append(target_idx_member)
    else:
        x0_member, edges_member, fixed_joints_member, motor_member, target_idx_member  = problem.convert_1D_to_mech(results.X)
        x0s.append(x0_member)
        edges.append(edges_member)
        fixed_joints.append(fixed_joints_member)
        motors.append(motor_member)
        target_idxs.append(target_idx_member)
save_figs(33)

# %% [markdown] cell 34
# here is how this differentiable class works:

# %% cell 35
distances, materials, distance_grads, material_grads = differentiable_optimization_tools(
    x0s,
    edges,
    fixed_joints,
    motors,
    target_curve,
    target_idxs
)

print(materials, distances)
save_figs(35)

# %% [markdown] cell 36
# **NOTE:** Sometimes the mechanisms are close to locking, which results in gradients exploding and in these cases the `DifferentiableTools` will return inifity results. You can ignore these since the `Tools` class will still simulate them eventhough the gradients are near singular.

# %% cell 37
x = x0s.copy()

step_size = 4e-4
n_steps = 1000

# keep track of which members are done optimizing
done_optimizing = np.zeros(len(x), dtype=bool)

x_last = x.copy()

for step in trange(n_steps):

    # get current distances, materials and gradients
    distances, materials, distance_grads, material_grads = differentiable_optimization_tools(
        x,
        edges,
        fixed_joints,
        motors,
        target_curve,
        target_idxs
    )

    # only update members that are valid and not done optimizing
    # (validity is judged against this target's own reference point)
    is_valid = np.logical_and(distances <= reference_point[0], materials <= reference_point[1])
    valids = np.where(is_valid)[0]
    invalids = np.where(~is_valid)[0]

    # if a member is invalid, revert to last step and mark as done optimizing
    for i in invalids:
        done_optimizing[i] = True
        x[i] = x_last[i]

    # keep a copy of last step
    x_last = x.copy()

    # update valid members
    for i in valids:
        if done_optimizing[i]:
            continue
        x[i] = x[i] - step_size * distance_grads[i]

    if np.all(done_optimizing):
        print(f'All members are done optimizing at step {step}')
        break
save_figs(37)

# %% [markdown] cell 38
# Now we can combine these new memebers with the old and see how the hypervolume changes:

# %% cell 39
combined_x0s = x0s + x
combined_edges = edges + edges
combined_fixed_joints = fixed_joints + fixed_joints
combined_motors = motors + motors
combined_target_idxs = target_idxs + target_idxs

F = np.array(
    PROBLEM_TOOLS(
        combined_x0s,
        combined_edges,
        combined_fixed_joints,
        combined_motors,
        target_curve,
        combined_target_idxs
    )).T

F_before = np.array(
    PROBLEM_TOOLS(
        x0s,
        edges,
        fixed_joints,
        motors,
        target_curve,
        target_idxs
    )).T

ref = np.array(reference_point)
ind = HV(ref)

hv_before = ind(F_before)
hv_after = ind(F)

print(f'Hypervolume before gradient optimization: {hv_before:.4f}, after optimization: {hv_after:.4f}')
save_figs(39)

# %% cell 40
plt.subplots(1, 2, figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Before Gradient Optimization')
ga_visualizer.plot_HV(F_before, ref, objective_labels=['Distance', 'Material'], ax=plt.gca())

plt.subplot(1, 2, 2)
plt.title('After Gradient Optimization')
ga_visualizer.plot_HV(F, ref, objective_labels=['Distance', 'Material'], ax=plt.gca())
save_figs(40)

# %% [markdown] cell 41
# A small improvement. But this is just to give you an idea of what you can do when mixing optimization methods. You can try different strategies to combine methods.
#
# Now you should have a good idea of how to use the tools we have provided to you. Think about how you can improve each aspect of the optimization to achieve a higher score in you final submission. You are allowed also submit your teams best solutions to the leaderboard which we will announce soon and see how well you are doing compared other teams.

# %% [markdown] cell 42
# A few hints and ideas:
# - Why not do some preprocessing to random mechanisms before running GA?
# - Can you use the gradients of both functions in gradient based optimization?
# - Can you cycle through multiple optimization runs?
# - Can you mix different kinds of GA? We showed how GA can be used for a single mechanism skeleton and how it can be used to also create the mechanisms?
# - Can you modify the GA mutations and cross-over?
# - Is there a more efficient representation of mechanisms we can use?
# - Are there smarter gradient-based optimization methods?

# %% [markdown] cell 43
# # Appendix and Details
#
# ## Curve Distance Calculation
#
# The target curves are provided to you as a sequence of points. The curves your mechanisms generate are then re-oriented compared to the target curves as we showed above. We measure the closeness of two point curves by computing the distance between the two curves with optimal alignment. This distance can be mathematically represented as:
#
# $$
# d_{OD}=\frac{2 \pi}{N}\min_{o_1 \in O_1, \alpha \in [0,2\pi)} \sum_{i=1}^N\left\|N(\mathbf{X}^{\text{coupler}}_{o_1(i)})\mathbf{R}(\alpha)-N(\mathbf{X}^{\text{target}})\right\|^2,
# $$
#
# $$\operatorname{N}(\mathbf{X})=\frac{\mathbf{X}- \overline{\mathbf{X}}}{\sqrt{\frac{\sum_{i=1}^N\|\mathbf{X}_i-\overline{\mathbf{X}}\|_2^2}{N}}}$$
#
# Where $\mathbf{X}^{\text{coupler}}$ is the curve traced by the mechanism and $\mathbf{X}^{\text{target}}$ is the target curve and  $O_1$ is the set of all possible clockwise and counterclockwise orderings of a curve with $N$ ordered points and $\mathbf{R}(\alpha)$ is a $2\times2$ rotation matrix for angle $\alpha$, while $o_1(i)$ refers to the index of the $i$-th point in the set $o_1$. In practice, in our problem we may ignore the denominator in $N$ since we do not want to optimally rescale curves. The code we provide solves this optimization problem and retuns the distance value so you don't have to worry about this.
#
# The other objective in this challenge problem is reducing the material used by the mechanism. This is the total length of all links in the mechanism. We have provided a function that makes this measurement for you using initial positions and adjacency matrix for a mechanism (The functions and useful tools are going to be described later).

# %% [markdown] cell 44
# ### A Note On Scale Normalization

# %% [markdown] cell 45
# The functions that we saw in the starter notebool on curves, namely `compare_curves` and `visualize_comparison`, and `visualize_alignment` have a scaling option. An important input to the `CurveEngine` class is `normalize_scale`, which if set to `True` will rescale curves when comparing. In our problem we want this to be `False` since we care about the scale and material used in the problem. (We stress this distinction because the first component of the optimization demo had this rescaling turned on!)

# %% cell 46
curve_processor = CurveEngine(
    normalize_scale=True, # whether to normalize the curve's scale (This would mean we do not care about the size of the curve but we do!)
    device='cpu' # device to run the processing on
)

curve_processor_unscaled = CurveEngine(
    normalize_scale=False, # do not normalize the scale of the curve (This would mean we care about the size of the curve too!)
    device='cpu'
)
save_figs(46)

# %% [markdown] cell 47
# Let's go back to the simple fourbar we had and see what this means:

# %% cell 48
visualizer = MechanismVisualizer()

# Define the variables for the mechanism
x0 = np.array([[0.3,0.2],
               [0.6,0.2],
               [0.3,0.3],
               [0.6,0.4],
               [0.4,0.5]])

# Define the connectivity of the mechanism
edges = np.array([[0,2],
                  [1,3],
                  [2,3],
                  [2,4],
                  [3,4]])

# Define the fixed nodes
fixed_joints = np.array([0,1])

# actuator
motor = np.array([0,2])

plt.figure(figsize=(5,5))
visualizer(x0, edges, fixed_joints, motor, ax=plt.gca())
save_figs(48)

# %% cell 49
solver = MechanismSolver(
    device = 'cpu' # device to run the simulation on (cpu or gpu)
)

solution = solver(x0, edges, fixed_joints, motor)
traced_curve = solution[4]
save_figs(49)

# %% cell 50
# lets compare to the first target curve (Kangaroo 1)
distance = curve_processor.compare_curves(traced_curve, target_curves[0])
unscaled_distance = curve_processor_unscaled.compare_curves(traced_curve, target_curves[0])

print(f"The scale normalized distance to {TARGET_NAMES[0]} is: {distance:.4f}")
print(f"The unnormalized distance to {TARGET_NAMES[0]} is: {unscaled_distance:.4f}") # This will be higher since do not normalize scale
save_figs(50)

# %% [markdown] cell 51
# Now let's visualize the alignment process with and without scale to understand what is going on:

# %% cell 52
curve_processor.visualize_alignment(traced_curve, target_curves[0])
curve_processor.visualize_comparison(traced_curve, target_curves[0])
save_figs(52)

# %% cell 53
curve_processor_unscaled.visualize_alignment(traced_curve, target_curves[0])
curve_processor_unscaled.visualize_comparison(traced_curve, target_curves[0])
save_figs(53)
