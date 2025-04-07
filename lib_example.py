import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH
import matplotlib.pyplot as plt
from spatialmath import SE3
from spatialmath.base import tr2rpy
from sklearn.metrics.pairwise import euclidean_distances
from final_class_hand_kinematics import Handkinematics as hk 

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
rf_cmc_config = [0,0,0,0,0,0,0,0,0,0]
rf_wcmc_config = [0,0,0,0,0,0,0,0]
lf_cmc_config = [0,0,0,0,0,0,0,0,0,0,0,0]
lf_wcmc_config = [0,0,0,0,0,0,0,0]
rf_common_cmc_config = [0,0,0,0,0,0,0,0,0,0,0]
lf_common_cmc_config = [0,0,0,0,0,0,0,0,0,0,0]

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

mf_params = [(0.085,0.0,0.0,np.pi/2,True,False),
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

rf_common_cmc_params = [(0.0,0.0,0.0,np.radians(62.7),True,False),
                (0.0316,np.pi/2,0.0,0.0,True,False),
                (0.0,np.pi/2,0.0,0.0,True,True),
                (0.049,0,0.0,-np.radians(13.83),True,False),
                (0.0,0,0.0,-np.radians(0),True,False),
                (0.0085,-np.pi/2,0.0,0.0,True,True),
                (0.04758,-np.pi/2,0.0,0.0,True,True),
                (0.0,np.pi/2,0.0,np.radians(7),True,False),
                (0.032175,-np.pi/2,0.0,0.0,True,True),
                (0.0,np.pi/2,0.0,np.radians(7),True,False),
                (0.020865,0.0,0.0,0.0,True,True)]

lf_common_cmc_params = [(0.0,0.0,0.0,np.radians(62.7),True,False),
                (0.0316,np.pi/2,0.0,0.0,True,False), 
                (0.0,np.pi/2,0.0,0.0,True,True),
                (0.0445,0,0.0,np.radians(14.25),True,False),
                (0.0,0.0,0.0,-np.radians(3),True,False),
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

proto3 = {'TH':(th_params, th_initial_config, TH_ROM),
    'IF': (if_params, if_inital_config, IF_ROM),
    'MF': (mf_params, mf_initial_config, MF_ROM),
    'RF': (rf_common_cmc_params, rf_common_cmc_config, RF_CMC_ROM),
    'LF': (lf_common_cmc_params, lf_common_cmc_config, LF_CMC_ROM)}

hand1 = hk(hand_info=proto1)
hand2 = hk(hand_info=proto2)
hand3 = hk(hand_info=proto3)

# Plotting the intial hand configuration
# fig1 = plt.figure()
# ax1 = plt.axes(projection="3d")
# hand1.plot_hand(ax=ax1)

# Compute the workspace for the prototype for each finger
hand1.compute_workspace()
hand2.compute_workspace()

# Plot the finger workspace for a prototype
# fig2 = plt.figure()
# ax2 = plt.axes(projection="3d")
# hand1.plot_finger_workspace(ax=ax2,finger='TH')
# plt.legend()

# Compute the workspace intersection between thumb and little finger
th_mdof, th_mdof_config, opp_mdof, opp_mdof_config = hand1.compute_workspace_intersection_points(thumb='TH',opposing_finger='LF')
th_pr2, th_pr2_config, opp_pr2, opp_pr2_config = hand2.compute_workspace_intersection_points(thumb='TH',opposing_finger='LF')
mdof_orientn_sim, mdof_orientn_euler = hand1.KOTcomputation(th_angles=th_mdof_config,oppfinger_angles=opp_mdof_config,oppfinger='LF')
proto2_orientn_sim, proto2_orientn_euler = hand2.KOTcomputation(th_angles=th_pr2_config,oppfinger_angles=opp_pr2_config,oppfinger='LF')

# Plot the finger workspace intersection for a prototype
hand1.plot_wksp_intersection(th_wksp=th_mdof,oppfinger_wksp=opp_mdof,config='MAX DOF')

# Comparing the prototypes for the KOT configuration

# Adjust these values to fine-tune the reference
xmin1 = 0.2
xmax1 = 0.3


xmin2 = 0.7
xmax2 = 0.8

meanprops = {
    'marker': '^',        
    'markerfacecolor': 'red',
    'markeredgecolor': 'black', 
    'markersize': 8
}

fig3 = plt.figure()
plt.boxplot([mdof_orientn_sim, proto2_orientn_sim], notch=False, patch_artist=True, showmeans=True, showfliers=False,
            boxprops=dict(facecolor='lightblue', color='blue'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='blue'),
            medianprops=dict(color='red'),
            meanprops=meanprops)
plt.axhline(y=2.82842712474619, color='k', linestyle='--', xmin=xmin1, xmax=xmax1, label='Desired Orientation Mismatch')
plt.axhline(y=2.82842712474619, color='k', linestyle='--', xmin=xmin2, xmax=xmax2)
plt.xticks([1, 2], ['MAX DOF', 'Prototype 2'])
plt.title('Variation of the orientation between thumb and opposing fingers fingertip')
plt.ylabel('Orientation mismatch')

# plt.savefig('Orientation Mismatch KOT.png', dpi=300)

# Animating the hand for one of the KOT poses

# Comment out the hand plotting codes to ensure that there are atmost two figure windows open for a clean animation, 
# when viewing the animation of the hand
# fig4 = plt.figure()
# ax3 = plt.axes(projection="3d")
# index1 = np.random.choice(th_mdof_config.shape[0])
# th_anim_config = th_mdof_config[index1,:]
# opp_anim_config = opp_mdof_config[index1,:]
# hand1.animate_hand(ax=ax3,th_jt_config=th_anim_config,opp_jt_config=opp_anim_config,oppfinger='LF')

# Cylinder surface parametrization
r = 0.03
theta = np.linspace(-np.pi,np.pi,30)
z = np.linspace(-0.075,0.075,30)
theta_prime = np.linspace(0,np.pi,30)
theta, z = np.meshgrid(theta, z)
x = r*np.cos(theta)
y = r*np.sin(theta)
cylinder_points = np.column_stack((x.ravel(),y.ravel(),z.ravel()))

# Cylinder transformation to lie diagonally in the palm of the hand
cylinder_transform = DHRobot([
    RevoluteDH(d=r, a=0.07*np.sin(np.pi/3), alpha=np.pi/2, offset=np.pi/3),
    RevoluteDH(d=-0.0435, a=0, alpha=0, offset=0),
])

T = cylinder_transform.fkine(q=[0,0])
rot_matrix = T.R
trans_matrix = T.t.reshape((3,1))
transformed_pts = rot_matrix @ cylinder_points.T + trans_matrix

# Filtering the cylinder points which are close to the cylinder curved surface area as candidate points for grasping
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

# Ensuring that each finger has the same number of candidate points to get an integer no. of grasps
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

# Plotting the candidate points for grasp optimization
# hand1.plot_wksp_pts_close_to_cylinder(close_pts=h1_close_pts, cylinder_pts=transformed_pts,config='MAX DOF')

# Visualizing the candidate grasps
def on_key(event):
    global proceed
    if event.key == 'n':  # Only continue on 'n' key
        proceed = True
        plt.close(event.canvas.figure)  # Close the current figure



for idx in range(4):

    proceed = False
    fig1 = plt.figure()
    ax1 = plt.axes(projection="3d")
    fig1.canvas.mpl_connect('key_press_event', on_key)

    lf_pt = h1_lf_pts[idx,:]
    rf_pt = h1_rf_pts[idx,:]
    mf_pt = h1_mf_pts[idx,:]
    if_pt = h1_if_candidate_pts[0][idx,:]
    th_pt = h1_th_pts[idx,:]

    th_config = h1_th_angles[idx,:]
    if_config = h1_if_candidate_pts[1][idx,:]
    mf_config = h1_mf_angles[idx,:]
    rf_config = h1_rf_angles[idx,:]
    lf_config = h1_lf_angles[idx,:]

    th_ip = hand1.hand_KC['TH'].fkine_all(th_config).t
    if_ip = hand1.hand_KC['IF'].fkine_all(if_config).t
    mf_ip = hand1.hand_KC['MF'].fkine_all(mf_config).t
    rf_ip = hand1.hand_KC['RF'].fkine_all(rf_config).t
    lf_ip = hand1.hand_KC['LF'].fkine_all(lf_config).t

    ax1.scatter(transformed_pts[0,:],transformed_pts[1,:],transformed_pts[2,:],c='cyan')
    ax1.scatter(lf_pt[0],lf_pt[1],lf_pt[2], c='olive',marker='o',label='LF')
    ax1.scatter(rf_pt[0],rf_pt[1],rf_pt[2], c='red',marker='o', label='RF')
    ax1.scatter(mf_pt[0],mf_pt[1],mf_pt[2], c='lime',marker='o', label='MF')
    ax1.scatter(if_pt[0],if_pt[1],if_pt[2], c='blue', marker='o', label='IF')
    ax1.scatter(th_pt[0],th_pt[1],th_pt[2], c='purple', marker='o', label='TH')
    ax1.plot(th_ip[:,0],th_ip[:,1],th_ip[:,2], c='purple', linewidth = 2, marker='x', label="Thumb")
    ax1.plot(lf_ip[:,0],lf_ip[:,1],lf_ip[:,2], c='olive', linewidth = 2, marker='x', label="LF")
    ax1.plot(if_ip[:,0],if_ip[:,1],if_ip[:,2],c='blue', linewidth = 2, marker='x',label="IF")
    ax1.plot(rf_ip[:,0],rf_ip[:,1],rf_ip[:,2],c='red', linewidth = 2, marker='x',label="RF")
    ax1.plot(mf_ip[:,0],mf_ip[:,1],mf_ip[:,2],c='lime', linewidth = 2, marker='x',label="MF")

    ax1.legend()
    plt.draw()
    plt.show()



    while not proceed:
        plt.pause(0.1)

plt.close()




plt.legend()
plt.show()