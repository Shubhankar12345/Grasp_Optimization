import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH
from scipy.spatial import ConvexHull
from spatialmath import SE3
from spatialmath.base import tr2rpy
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import block_diag
from scipy import optimize
from final_class_hand_kinematics import Handkinematics as hk

def skew_symm_operator(r: np.ndarray):

    S = np.array([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]])

    return S

def get_contact_frame(axis,center,appl_pt):

    t_hat = axis/np.linalg.norm(axis)

    r = appl_pt - center
    
    r_proj = np.dot(r,t_hat)*t_hat

    n = r - r_proj

    n_hat = -n/np.linalg.norm(n)

    s_hat = np.cross(n_hat, t_hat)
    
    s_hat /= np.linalg.norm(s_hat)
 
    return n_hat, s_hat, t_hat

def partial_grasp_matrix(orientation: np.ndarray, ci: np.ndarray, obj_centroid: np.ndarray):

    P = np.block([[np.eye(3), np.zeros((3,3))],[skew_symm_operator(r=ci-obj_centroid), np.eye(3)]])

    R = np.block([[orientation, np.zeros((3,3))],[np.zeros((3,3)), orientation]])

    Gt = R.T @ P.T

    return Gt

# Function to plot the convex hull
def plot_convex_hull(points, color='cyan', alpha=0.3):
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        simplex = np.append(simplex, simplex[0])  # Close the loop
        ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], color=color)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', s=5)

# Grasp matrix G and friction cone generators S must be given
# Example placeholders (replace with your actual G and S)
# G shape = (6, nc*3)
# S shape = (nc*3, ke), ke = number of friction cone generators per contact

def compute_GWS(G, S):
    """
    Compute grasp wrench space as the wrenches obtained by multiplying G and S.
    """
    # G is (6, nc*3)
    # S is (nc*3, ke)
    W = G @ S  # (6, ke)
    return W

# === Example Inputs === #
# Let's assume 5 contacts, 3 force components per contact (nc*3 = 15)
nc = 5
ke = 16  # 16 friction cone generators
mu_EPDM = 1
N_sample = 5000
np.random.seed(0)

TH_ROM = {'TH_CMC_a': np.linspace(-np.radians(92),np.radians(28),N_sample),
      'TH_CMC_f': np.linspace(-np.radians(15),np.radians(60),N_sample),
      'TH_MCP_a': np.linspace(-np.radians(30),0,N_sample),
      'TH_MCP_f': np.linspace(-np.radians(55),0,N_sample),
      'TH_IP': np.linspace(-np.radians(80),0,N_sample)}

IF_ROM = {'IF_MCP_a': np.linspace(-np.pi/6,np.pi/6,N_sample),
      'IF_MCP_f': np.linspace(0,(4*np.pi)/9,N_sample),
      'IF_PIP': np.linspace(0,(5*np.pi)/9,N_sample),
      'IF_DIP': np.linspace(0,np.pi/2,N_sample)}

MF_ROM = {'MF_MCP_a': np.linspace(-np.radians(22.5),np.radians(22.5),N_sample),
      'MF_MCP_f': np.linspace(0,np.radians(80),N_sample),
      'MF_PIP': np.linspace(0,np.radians(100),N_sample),
      'MF_DIP': np.linspace(0,np.pi/2,N_sample)}

RF_CMC_ROM = {'RF_CMC': np.linspace(0,np.radians(10),N_sample),
    'RF_MCP_a': np.linspace(-np.radians(22.5),np.radians(22.5),N_sample),
    'RF_MCP_f': np.linspace(0,np.radians(80),N_sample),
    'RF_PIP': np.linspace(0,np.radians(100),N_sample),
    'RF_DIP': np.linspace(0,np.pi/2,N_sample)}

RF_WCMC_ROM = {'RF_MCP_a': np.linspace(-np.radians(22.5),np.radians(22.5),N_sample),
    'RF_MCP_f': np.linspace(0,np.radians(80),N_sample),
    'RF_PIP': np.linspace(0,np.radians(100),N_sample),
    'RF_DIP': np.linspace(0,np.pi/2,N_sample)}

LF_CMC_ROM = {'LF_CMC': np.linspace(0,np.radians(20),N_sample),
    'LF_MCP_a': np.linspace(-np.radians(25),np.radians(25),N_sample),
    'LF_MCP_f': np.linspace(0,np.radians(80),N_sample),
    'LF_PIP': np.linspace(0,np.radians(100),N_sample),
    'LF_DIP': np.linspace(0,np.pi/2,N_sample)}

LF_WCMC_ROM = {'LF_MCP_a': np.linspace(-np.radians(25),np.radians(25),N_sample),
    'LF_MCP_f': np.linspace(0,np.radians(80),N_sample),
    'LF_PIP': np.linspace(0,np.radians(100),N_sample),
    'LF_DIP': np.linspace(0,np.pi/2,N_sample)}

th_initial_config = [0,-np.radians(32),0,0,0,0,0]
if_inital_config = [0,0,0,0,0,0,0]
mf_initial_config = [0,0,0,0,0]
rf_cmc_config = [0,0,0,0,0,0,0,0,0]
rf_wcmc_config = [0,0,0,0,0,0,0,0]
lf_cmc_config = [0,0,0,0,0,0,0,0,0]
lf_wcmc_config = [0,0,0,0,0,0,0,0]

th_params = [(0.028,0,0.02,np.radians(153.5),True,False),
            (0.0,np.pi/2,0.0,0.0,True,True),
            (0.0,np.pi/2,0,0,True,True),
            (0.048945,-np.radians(60),0,np.radians(16.5),True,False),
            (0.0075,-np.pi/2,0.0,0.0,True,True),
            (0.03822,0.0,0.0,0.0,True,True),
            (0.03081,0.0,0.0,0.0,True,True)]

if_params = [(0.0,0.0,0.0,np.radians(106.5),True,False),
            (0.088,0.0,0.0,0.0,True,False), 
            (0.0085,0.0,0.0,0.0,True,True),
            (0.0,np.pi/2,0.0,-np.radians(10),True,False),
            (0.047775,0.0,0.0,0.0,True,True),
            (0.027885,0,0,0,True,True),
            (0.018915,0,0,0,True,True)]

mf_params = [(0.087,0.0,0.0,np.pi/2,True,False),
            (0.0085,np.pi/2,0.0,0.0,True,True),
            (0.05187,0.0,0.0,0.0,True,True),
            (0.03315,0.0,0.0,0.0,True,True),
            (0.018915,0.0,0.0,0.0,True,True)]

rf_mdof_params = [(0.0,0.0,0.0,np.radians(72.5),True,False),
                (0.031,np.pi/2,0.0,0.0,True,False),
                (0.0,np.pi/2,0.0,0.0,True,True),
                (0.0485,0,0.0,-np.radians(4),True,False),
                (0.0075,-np.pi/2,0.0,0.0,True,True),
                (0.04758,-np.pi/2,0.0,0.0,True,True),
                (0.0,np.pi/2,0.0,np.radians(7),True,False),
                (0.032175,-np.pi/2,0.0,0.0,True,True),
                (0.0,np.pi/2,0.0,np.radians(7),True,False),
                (0.020865,0.0,0.0,0.0,True,True)]

rf_wcmc_params = [(0.0,0.0,0.0,np.radians(72.5),True,False),
                (0.0795,np.pi,0.0,0.0,True,False),
                (0.0085,-np.pi/2,0.0,0.0,True,True),
                (0.04758,-np.pi/2,0.0,0.0,True,True),
                (0.0,np.pi/2,0.0,np.radians(7),True,False),
                (0.032175,-np.pi/2,0.0,0.0,True,True),
                (0.0,np.pi/2,0.0,np.radians(7),True,False),
                (0.020865,0.0,0.0,0.0,True,True)]

lf_mdof_params = [(0.0,0.0,0.0,np.radians(53.5),True,False),
                (0.033,np.pi/2,0.0,0.0,True,False), 
                (0.0,np.pi/2,0.0,0.0,True,True),
                (0.043,0.0,0.0,-np.radians(7.9),True,False),
                (0.0,0.0,0.0,np.radians(4),True,False),
                (0.0,np.pi/2,0.0,0.0,True,True),
                (0.0075,np.pi,0,0,True,False),
                (0.03978,-np.pi/2,0.0,0.0,True,True),
                (0.0,np.pi/2,0.0,np.radians(10),True,False),
                (0.022815,-np.pi/2,0.0,0.0,True,True),
                (0.0,np.pi/2,0.0,np.radians(10),True,False),
                (0.018135,0.0,0.0,0.0,True,True)]

lf_wcmc_params = [(0.0,0.0,0.0,np.radians(63.5),True,False),
                (0.076,np.pi,0.0,0.0,True,False),
                (0.0085,-np.pi/2,0.0,0.0,True,True),
                (0.03978,-np.pi/2,0.0,0.0,True,True),
                (0.0,np.pi/2,0.0,np.radians(10),True,False),
                (0.022815,-np.pi/2,0.0,0.0,True,True),
                (0.0,np.pi/2,0.0,np.radians(10),True,False),
                (0.018135,0.0,0.0,0.0,True,True)]

proto1 = {'TH':(th_params, th_initial_config, TH_ROM),
    'IF': (if_params, if_inital_config, IF_ROM),
    'MF': (mf_params, mf_initial_config, MF_ROM),
    'RF': (rf_mdof_params, rf_cmc_config, RF_CMC_ROM),
    'LF': (lf_mdof_params, lf_cmc_config, LF_CMC_ROM)}

proto2 = {'TH':(th_params, th_initial_config, TH_ROM),
    'IF': (if_params, if_inital_config, IF_ROM),
    'MF': (mf_params, mf_initial_config, MF_ROM),
    'RF': (rf_wcmc_params, rf_wcmc_config, RF_WCMC_ROM),
    'LF': (lf_wcmc_params, lf_wcmc_config, LF_WCMC_ROM)}

epsilon = 1e-6
hand1 = hk(hand_info=proto1)
hand1.compute_workspace()
th_pts, th_jt_config, opp_pts, opp_jt_config = hand1.compute_workspace_intersection_points(thumb='TH',opposing_finger='LF')

# Cylinder surface parametrization
r = 0.03
theta = np.linspace(-np.pi,np.pi,30)
z = np.linspace(-0.075,0.075,30)

theta, z = np.meshgrid(theta, z)
x = r*np.cos(theta)
y = r*np.sin(theta)
cylinder_points = np.column_stack((x.ravel(),y.ravel(),z.ravel()))

cylinder_transform = DHRobot([
    RevoluteDH(d=r, a=0.07*np.sin(np.pi/3), alpha=np.pi/2, offset=np.pi/3),
    RevoluteDH(d=-0.0435, a=0, alpha=0, offset=0),
])
T = cylinder_transform.fkine(q=[0,0])
rot_matrix = T.R
trans_matrix = T.t.reshape((3,1))
transformed_pts = rot_matrix @ cylinder_points.T + trans_matrix

h1_th_candidate_pts = hand1.filter_points_near_cylinder(finger='TH',cylinder_points=transformed_pts.T,
                    cylinder_axis=rot_matrix[:,2],cylinder_upward_axis=rot_matrix[:,1],cylinder_radius=r,
                    cylinder_lowerlimit=np.min(z),cylinder_upper_limit=np.max(z)/3)
h1_if_candidate_pts = hand1.filter_points_near_cylinder(finger='IF',cylinder_points=transformed_pts.T,
                    cylinder_axis=rot_matrix[:,2],cylinder_upward_axis=rot_matrix[:,1],cylinder_radius=r,
                    cylinder_lowerlimit=np.min(z),cylinder_upper_limit=0)
h1_mf_candidate_pts = hand1.filter_points_near_cylinder(finger='MF',cylinder_points=transformed_pts.T,
                    cylinder_axis=rot_matrix[:,2],cylinder_upward_axis=rot_matrix[:,1],cylinder_radius=r,
                    cylinder_lowerlimit=np.min(z)/6,cylinder_upper_limit=np.max(z)/15)
h1_rf_candidate_pts = hand1.filter_points_near_cylinder(finger='RF',cylinder_points=transformed_pts.T,
                    cylinder_axis=rot_matrix[:,2],cylinder_upward_axis=rot_matrix[:,1],cylinder_radius=r,
                    cylinder_lowerlimit=0,cylinder_upper_limit=np.max(z)/2)
h1_lf_candidate_pts = hand1.filter_points_near_cylinder(finger='LF',cylinder_points=transformed_pts.T,
                    cylinder_axis=rot_matrix[:,2],cylinder_upward_axis=rot_matrix[:,1],cylinder_radius=r,
                    cylinder_lowerlimit=0,cylinder_upper_limit=np.max(z))

print(h1_th_candidate_pts[0].shape, h1_if_candidate_pts[0].shape,h1_mf_candidate_pts[0].shape,h1_rf_candidate_pts[0].shape,h1_lf_candidate_pts[0].shape)

indices = np.random.choice(h1_lf_candidate_pts[0].shape[0],h1_th_candidate_pts[0].shape[0],replace=False)

h1_th_pts = h1_th_candidate_pts[0]
h1_if_pts = h1_if_candidate_pts[0][indices]
h1_mf_pts = h1_mf_candidate_pts[0][indices]
h1_rf_pts = h1_rf_candidate_pts[0][indices]
h1_lf_pts = h1_lf_candidate_pts[0][indices]

h1_th_angles = h1_th_candidate_pts[1]
h1_if_angles = h1_if_candidate_pts[1][indices]
h1_mf_angles = h1_mf_candidate_pts[1][indices]
h1_rf_angles = h1_rf_candidate_pts[1][indices]
h1_lf_angles = h1_lf_candidate_pts[1][indices]

h1_close_pts = (h1_th_pts,h1_if_pts,h1_mf_pts,h1_rf_pts,h1_lf_pts)

cylinder_center = np.mean(transformed_pts,axis=1)

# Placeholder G and S (replace with your actual ones)
for i in range(146):
    n_l,s_l,t_l = get_contact_frame(axis=rot_matrix[:,2],center=cylinder_center,appl_pt=h1_lf_pts[i,:])
    R_l = np.column_stack((n_l,t_l,s_l))
    n_r,s_r,t_r = get_contact_frame(axis=rot_matrix[:,2],center=cylinder_center,appl_pt=h1_rf_pts[i,:])
    R_r = np.column_stack((n_r,t_r,s_r))
    n_m,s_m,t_m = get_contact_frame(axis=rot_matrix[:,2],center=cylinder_center,appl_pt=h1_mf_pts[i,:])
    R_m = np.column_stack((n_m,t_m,s_m))
    n_i,s_i,t_i = get_contact_frame(axis=rot_matrix[:,2],center=cylinder_center,appl_pt=h1_if_pts[i,:])
    R_i = np.column_stack((n_i,t_i,s_i))
    n_th,s_th,t_th = get_contact_frame(axis=rot_matrix[:,2],center=cylinder_center,appl_pt=h1_th_pts[i,:])
    R_th = np.column_stack((n_th,t_th,s_th))

    G5 = partial_grasp_matrix(orientation=R_l,ci=h1_lf_pts[i,:],obj_centroid=cylinder_center)
    G4 = partial_grasp_matrix(orientation=R_r,ci=h1_rf_pts[i,:],obj_centroid=cylinder_center)
    G3 = partial_grasp_matrix(orientation=R_m,ci=h1_mf_pts[i,:],obj_centroid=cylinder_center)
    G2 = partial_grasp_matrix(orientation=R_i,ci=h1_if_pts[i,:],obj_centroid=cylinder_center)
    G1 = partial_grasp_matrix(orientation=R_th,ci=h1_th_pts[i,:],obj_centroid=cylinder_center)

    F = 1*rot_matrix[:,2].reshape(3,1)
    w_ext = np.squeeze(np.vstack((F,np.zeros((3,1)))),axis=1)

    G_tilda_T = np.vstack((G1,G2,G3,G4,G5))
    H_i = np.hstack([np.eye(3), np.zeros((3, 3))])
    H = block_diag(*(H_i for _ in range(5)))
    G_T = H @ G_tilda_T
    G = G_T.T
    S_list = []
    for i in range(nc):
        Si = np.zeros((3, ke))
        for j in range(ke):
            theta = 2 * np.pi * j / ke
            Si[:, j] = np.array([1, mu_EPDM * np.cos(theta), mu_EPDM * np.sin(theta)])
        S_list.append(Si)

    S_block_diag = np.block([
        [S_list[i] if i == j else np.zeros((3, ke)) for j in range(nc)]
        for i in range(nc)
    ])

    # === Compute Grasp Wrenches === #
    W = compute_GWS(G, S_block_diag)  # W is (6, ke)

    # W is of shape (6, ke), you need its transpose to multiply by lambda
    W_matrix = W.T  # Shape: (ke, 6)

    # Objective: We only care about feasibility, so we can minimize 0
    c = np.zeros(W_matrix.shape[0])

    # Equality constraint: W.T @ lambda = w_ext
    A_eq = W_matrix.T  # Shape: (6, ke)
    b_eq = -w_ext       # Shape: (6,)

    # Inequality constraint: lambda >= 0
    bounds = [(0, None) for _ in range(W_matrix.shape[0])]

    # Solve LP
    res = optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success:
        print("w_ext is inside the Grasp Wrench Space.")
    else:
        print("w_ext is NOT inside the Grasp Wrench Space.")

quit()

# For visualization, we'll only look at the first 3 components (forces)
forces = W[:3, :].T  # Shape (ke, 3)

# === Plot the Grasp Wrench Space === #
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot convex hull of wrench space
plot_convex_hull(forces, color='red')
ax.scatter(w_ext[0], w_ext[1], w_ext[2], color='green', s=50, label='External Wrench')

# Labels and visualization tweaks
ax.set_xlabel('Force X')
ax.set_ylabel('Force Y')
ax.set_zlabel('Force Z')
ax.set_title('Grasp Wrench Space (Forces Only)')

plt.show()