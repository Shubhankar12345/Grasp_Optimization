import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH
import matplotlib.pyplot as plt
from spatialmath import SE3
from spatialmath.base import tr2rpy
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import block_diag
from scipy import optimize
from final_class_hand_kinematics import Handkinematics as hk 

def skew_symm_operator(r: np.ndarray):
    """
    Generates the skew symmetric matrix S(r) for the cross product operation for a 3D vector r

    This matrix can be used to compute the cross product as a matrix multiplication:
    S(r) @ v == np.cross(r, v)

    Args:
        r (np.ndarray): 1D array of shape (3,), representing the vector from the object's 
        centroid to the finger contact location 

    Returns:
        np.ndarray: The 3x3 skew symmetric matrix for the cross product operation

    Example:
        r = np.array([1, 2, 3])
        print(skew_symm_operator(r))
        array([[ 0, -3,  2],
               [ 3,  0, -1],
               [-2,  1,  0]])
    """

    S = np.array([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]])

    return S

def get_contact_frame(axis: np.ndarray,center: np.ndarray,appl_pt: np.ndarray):
    """

    Defines a right-handed local contact frame {n_hat,s_hat,t_hat} at the finger contact point at the object

    Args:
        axis (np.ndarray): 1D array of shape (3,), describing the cylinder axis vector in the global frame, 
        i.e. the wrist reference frame
        center (np.ndarray): 1D array of shape (3,), describing the cylinder center in the global frame,
        i.e. the wrist reference frame
        appl_pt (np.ndarray): 1D array of shape (3,), describing the finger contact point in the global frame,
        i.e. the wrist reference frame 

    Returns:
        n_hat (np.ndarray): 1D array of shape (3,), describing the inward unit normal at that contact point
        s_hat (np.ndarray): 1D array of shape (3,), describing the unit sliding vector in the tangent plane,
        at that contact point
        t_hat (np.ndarray): 1D array of shape (3,), describing the unit tangential vector in the tangent plane,
        at that contact point as s_hat = n_hat x t_hat

    Examples:
        ax = np.array([0,1,0])
        cent = np.array([1,2,3])
        ap_pt = np.array([0.2,0.5,1.3])

        n,s,t = get_contact_frame(axis=ax,center=cent,appl_pt = ap_pt)
    """
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

def feasibility_objective(f):
    residual = G @ f + w_ext
    return 0.5*np.dot(residual, residual)  # ||Gf + w_ext||^2

def feasibility_objective_grad(f):
    residual = G @ f + w_ext
    grad = G.T @ residual
    return grad

def objective(f: np.ndarray):
    return 0.5*np.dot(f.T,f) + 1e-6 * np.linalg.norm(f)

def objective_jac(f: np.ndarray):
    return f + 1e-6 * (f / np.linalg.norm(f))

def eq_fn(f: np.ndarray, n_contacts: np.int32):
    residual = (G@f + w_ext).flatten()
    return residual

def eq_jac(f: np.ndarray, n_contacts: np.int32):
    jac_mat = G
    return jac_mat

def nonlin_cons_eq(f: np.ndarray, n_contacts: np.int32):
    nonlin_cons = np.zeros(n_contacts)
    mu_EPDM = 1
    for i in range(n_contacts):
        fi = f[3*i:3*i+3]
        fn = fi[0]  # normal
        fs = fi[1]  # shear/sliding
        ft = fi[2]  # tangential
        nonlin_cons[i] = mu_EPDM * fn - np.sqrt(fs**2 + ft**2)
    return nonlin_cons

def nonlin_cons_jac(f: np.ndarray, n_contacts: np.int32):
    nonlin_cons_jac_mat = np.zeros((n_contacts, 3*n_contacts))
    mu_EPDM = 1
    epsilon = 1e-12
    for i in range(n_contacts):
        fi = f[3*i:3*i+3]
        fn = fi[0]
        fs = fi[1]
        ft = fi[2]
        magnitude = np.sqrt(fs**2 + ft**2)
        if magnitude > epsilon:
            jac_row = np.array([
                mu_EPDM,           # ∂/∂f_n
                -fs / magnitude,   # ∂/∂f_s
                -ft / magnitude    # ∂/∂f_t
            ])
        else:
            jac_row = np.array([mu_EPDM, 0.0, 0.0])
        nonlin_cons_jac_mat[i, 3*i:3*i+3] = jac_row
    return nonlin_cons_jac_mat

def nonlin_cons_hessian(f: np.ndarray, v: np.ndarray, n_contacts: np.int32):
    nonlin_cons_hessian = np.zeros((3*n_contacts,3*n_contacts))
    mu_EPDM = 1
    epsilon = 1e-12
    for i in range(n_contacts):
        fi = f[3*i:3*i+3]
        fi = f[3*i:3*i+3]
        fn = fi[0]
        fs = fi[1]
        ft = fi[2]
        magnitude = np.sqrt(fs**2 + ft**2)
        if(magnitude > epsilon):
            H_i = np.array([[0,0,0],
                            [0,-(ft**2)/(magnitude**3),(fs*ft)/(magnitude**3)],
                            [0,(fs*ft)/(magnitude**3),-(fs**2)/magnitude**3]])
        else:
            H_i = np.zeros((3,3))
                    
        nonlin_cons_hessian[3*i:3*i+3,3*i:3*i+3] = v[i]*H_i

    return nonlin_cons_hessian

np.random.seed(0)
N_sample = 5000

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
rf_cmc_config = [0,0,0,0,0,0,0,0,0,0]
rf_wcmc_config = [0,0,0,0,0,0,0,0]
lf_cmc_config = [0,0,0,0,0,0,0,0,0,0,0,0]
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
idx_sz = len(indices)

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

F = 1*rot_matrix[:,2].reshape(3,1)
w_ext = np.squeeze(np.vstack((F,np.zeros((3,1)))),axis=1)
mu_EPDM = 1

cylinder_center = np.mean(transformed_pts,axis=1)

for i in range(idx_sz):
    n_l,s_l,t_l = get_contact_frame(axis=rot_matrix[:,2],center=cylinder_center,appl_pt=h1_lf_pts[i,:])
    R_l = np.array([n_l,t_l,s_l])
    n_r,s_r,t_r = get_contact_frame(axis=rot_matrix[:,2],center=cylinder_center,appl_pt=h1_rf_pts[i,:])
    R_r = np.array([n_r,t_r,s_r])
    n_m,s_m,t_m = get_contact_frame(axis=rot_matrix[:,2],center=cylinder_center,appl_pt=h1_mf_pts[i,:])
    R_m = np.array([n_m,t_m,s_m])
    n_i,s_i,t_i = get_contact_frame(axis=rot_matrix[:,2],center=cylinder_center,appl_pt=h1_if_pts[i,:])
    R_i = np.array([n_i,t_i,s_i])
    n_th,s_th,t_th = get_contact_frame(axis=rot_matrix[:,2],center=cylinder_center,appl_pt=h1_th_pts[i,:])
    R_th = np.array([n_th,t_th,s_th])

    G5 = partial_grasp_matrix(orientation=R_l,ci=h1_lf_pts[i,:],obj_centroid=cylinder_center)
    G4 = partial_grasp_matrix(orientation=R_r,ci=h1_rf_pts[i,:],obj_centroid=cylinder_center)
    G3 = partial_grasp_matrix(orientation=R_m,ci=h1_mf_pts[i,:],obj_centroid=cylinder_center)
    G2 = partial_grasp_matrix(orientation=R_i,ci=h1_if_pts[i,:],obj_centroid=cylinder_center)
    G1 = partial_grasp_matrix(orientation=R_th,ci=h1_th_pts[i,:],obj_centroid=cylinder_center)

    G_tilda_T = np.vstack((G1,G2,G3,G4,G5))
    H = block_diag(np.eye(3),np.eye(3),np.eye(3),np.eye(3),np.eye(3))
    H1 = block_diag(np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3)))
    H = np.hstack((H,H1))
    G_T = H @ G_tilda_T
    G = G_T.T

    lb = (-10)*np.ones(15)
    lb[::3] = 1e-2
    ub = 10*np.ones(15)
    bounds = optimize.Bounds(lb,ub)

    linear_constraint = optimize.LinearConstraint(G, -w_ext-epsilon, -w_ext+epsilon)
    nonlinear_constraint = optimize.NonlinearConstraint(fun=lambda f: nonlin_cons_eq(f, n_contacts=5), lb=-1e-3, ub=np.inf, jac=lambda f: nonlin_cons_jac(f, n_contacts=5), hess= lambda f,v: nonlin_cons_hessian(f, v, n_contacts=5))
    f0 = np.linalg.pinv(G)@(-w_ext)

    for j in range(5):
        fn = f0[3*j]
        if fn < 1e-2:
            f0[3*j] = 1e-2  # ensure positive normal force
        fs = f0[3*j+1]
        ft = f0[3*j+2]
        tang_mag = np.sqrt(fs**2 + ft**2)
        mu_fn = mu_EPDM * f0[3*j]
        if tang_mag > mu_fn:
            scale = mu_fn / (tang_mag + 1e-9)
            f0[3*j+1] *= scale
            f0[3*j+2] *= scale

    res_phase1 = res_phase1 = optimize.minimize(
    fun=feasibility_objective,
    jac=feasibility_objective_grad,
    constraints=[nonlinear_constraint],
    x0=f0,
    method='trust-constr',
    bounds=bounds,
    options={'gtol': 1e-6, 'xtol': 1e-6, 'barrier_tol': 1e-6})
    f_feasible = res_phase1.x

    print("Initial Residual (equilibrium):", G @ f_feasible + w_ext)
    print("Initial Friction constraints:", nonlin_cons_eq(f_feasible, 5))

    res = optimize.minimize(fun=objective,x0=f_feasible, method='trust-constr', jac=objective_jac,
                constraints=[linear_constraint, nonlinear_constraint],
                options={'xtol': 1e-6, 'gtol': 1e-6, 'barrier_tol': 1e-6, 'maxiter': 2000}, bounds=bounds)
    fres = res.x

    print(fres)
    print("Residual (equilibrium):", G @ fres + w_ext)
    print("Friction constraints:", nonlin_cons_eq(fres, 5))
    print("Normal force mins:", fres[::3])
    print(f"Iteration {i}")
    print("-----------------------------------------------------------------------------------------------")

hand1.plot_wksp_pts_close_to_cylinder(close_pts=h1_close_pts, cylinder_pts=transformed_pts,config='MAX DOF')
plt.show()