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
lf_cmc_config = [0,0,0,0,0,0,0,0,0,0,0,0]

th_params = [(0.028,0,0.02,np.radians(153.5),True,False),
            (0.0,np.pi/2,0.0,0.0,True,True),
            (0.0,np.pi/2,0,0,True,True),
            (0.048945,-np.radians(60),0,np.radians(16.5),True,False),
            (0.0075,-np.pi/2,0.0,0.0,True,True),
            (0.03822,0.0,0.0,0.0,True,True),
            (0.03081,0.0,0.0,0.0,True,True)]

if_params = [(0.0,0.0,0.0,np.radians(106.5),True,False),
            (0.088,0.0,0.0,0.0,True,False), 
            (0.0075,0.0,0.0,0.0,True,True),
            (0.0,np.pi/2,0.0,-np.radians(10),True,False),
            (0.047775,0.0,0.0,0.0,True,True),
            (0.027885,0,0,0,True,True),
            (0.018915,0,0,0,True,True)]

mf_params = [(0.085,0.0,0.0,np.pi/2,True,False),
            (0.0075,np.pi/2,0.0,0.0,True,True),
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

proto1 = {'TH':(th_params, th_initial_config, TH_ROM),
    'IF': (if_params, if_inital_config, IF_ROM),
    'MF': (mf_params, mf_initial_config, MF_ROM),
    'RF': (rf_mdof_params, rf_cmc_config, RF_CMC_ROM),
    'LF': (lf_mdof_params, lf_cmc_config, LF_CMC_ROM)}

# fig1 = plt.figure()
# ax1 = plt.axes(projection="3d")
hand1 = hk(hand_info=proto1)
# hand1.plot_hand(ax=ax1)
hand1.compute_workspace()

# fig2 = plt.figure(figsize=(5,5))
# ax2 = plt.axes(projection="3d")
# hand1.plot_finger_workspace(ax=ax2,finger='TH')
# hand1.plot_finger_workspace(ax=ax2,finger='RF')
# hand1.plot_finger_workspace(ax=ax2,finger='LF')

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
cylinder_center = np.mean(transformed_pts,axis=1)
cylinder_axis = rot_matrix[:,2]
# ax2.scatter(transformed_pts[0,:],transformed_pts[1,:],transformed_pts[2,:])



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
# hand1.plot_wksp_pts_close_to_cylinder(close_pts=h1_close_pts, cylinder_pts=transformed_pts,config='MAX DOF')
# plt.show()

# def on_key(event):
#     global proceed
#     if event.key == 'n':  # Only continue on 'n' key
#         proceed = True
#         plt.close(event.canvas.figure)  # Close the current figure



# for idx in range(4):

#     proceed = False
#     fig1 = plt.figure()
#     ax1 = plt.axes(projection="3d")
#     fig1.canvas.mpl_connect('key_press_event', on_key)

#     lf_pt = h1_lf_pts[idx,:]
#     rf_pt = h1_rf_pts[idx,:]
#     mf_pt = h1_mf_pts[idx,:]
#     if_pt = h1_if_candidate_pts[0][idx,:]
#     th_pt = h1_th_pts[idx,:]

#     th_config = h1_th_angles[idx,:]
#     if_config = h1_if_candidate_pts[1][idx,:]
#     mf_config = h1_mf_angles[idx,:]
#     rf_config = h1_rf_angles[idx,:]
#     lf_config = h1_lf_angles[idx,:]

#     th_ip = hand1.hand_KC['TH'].fkine_all(th_config).t
#     if_ip = hand1.hand_KC['IF'].fkine_all(if_config).t
#     mf_ip = hand1.hand_KC['MF'].fkine_all(mf_config).t
#     rf_ip = hand1.hand_KC['RF'].fkine_all(rf_config).t
#     lf_ip = hand1.hand_KC['LF'].fkine_all(lf_config).t

#     ax1.scatter(transformed_pts[0,:],transformed_pts[1,:],transformed_pts[2,:],c='cyan')
#     ax1.scatter(cylinder_center[0],cylinder_center[1],cylinder_center[2],c='black')
#     ax1.scatter(lf_pt[0],lf_pt[1],lf_pt[2], c='olive',marker='o',label='LF')
#     ax1.scatter(rf_pt[0],rf_pt[1],rf_pt[2], c='red',marker='o', label='RF')
#     ax1.scatter(mf_pt[0],mf_pt[1],mf_pt[2], c='lime',marker='o', label='MF')
#     ax1.scatter(if_pt[0],if_pt[1],if_pt[2], c='blue', marker='o', label='IF')
#     ax1.scatter(th_pt[0],th_pt[1],th_pt[2], c='purple', marker='o', label='TH')
#     ax1.plot(th_ip[:,0],th_ip[:,1],th_ip[:,2], c='purple', linewidth = 2, marker='x', label="Thumb")
#     ax1.plot(lf_ip[:,0],lf_ip[:,1],lf_ip[:,2], c='olive', linewidth = 2, marker='x', label="LF")
#     ax1.plot(if_ip[:,0],if_ip[:,1],if_ip[:,2],c='blue', linewidth = 2, marker='x',label="IF")
#     ax1.plot(rf_ip[:,0],rf_ip[:,1],rf_ip[:,2],c='red', linewidth = 2, marker='x',label="RF")
#     ax1.plot(mf_ip[:,0],mf_ip[:,1],mf_ip[:,2],c='lime', linewidth = 2, marker='x',label="MF")

#     ax1.legend()
#     plt.draw()
#     plt.show()



#     while not proceed:
#         plt.pause(0.1)

# plt.close()



th_ip = hand1.hand_KC['TH'].fkine_all(th_initial_config).t
fig2 = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlim((-0.05,0.1))
ax.set_ylim((-0.02,0.2))
ax.set_zlim((0,0.025))
ax.view_init(elev = 30, azim = -45)
ax.plot(th_ip[:,0],th_ip[:,1],th_ip[:,2], c='purple', marker='x', label="Thumb")
th = hand1.hand_KC['TH'].fkine_all(th_initial_config).A
# T23 = np.linalg.inv(th[6])@th[7]
# print(T23)
arrow_length = 0.005
for i in range(7):
    th_frame = th[i]
    th_origin = th_frame[0:3, 3]
    th_x_axis = th_frame[0:3, 0]
    th_y_axis = th_frame[0:3, 1]
    th_z_axis = th_frame[0:3, 2]
    ax.quiver(th_origin[0], th_origin[1], th_origin[2], 
            th_x_axis[0], th_x_axis[1], th_x_axis[2], 
            color='red', length=arrow_length, linewidth=2)
    ax.quiver(th_origin[0], th_origin[1], th_origin[2], 
            th_z_axis[0], th_z_axis[1], th_z_axis[2], 
            color='blue', length=arrow_length, linewidth=2)
th_frame = th[-1]
th_origin = th_frame[0:3, 3]
th_x_axis = th_frame[0:3, 0]
th_y_axis = th_frame[0:3, 1]
th_z_axis = th_frame[0:3, 2]
ax.quiver(th_origin[0], th_origin[1], th_origin[2], 
        th_x_axis[0], th_x_axis[1], th_x_axis[2], 
        color='red', length=arrow_length, label='X-axis', linewidth=2)
ax.quiver(th_origin[0], th_origin[1], th_origin[2], 
        th_y_axis[0], th_y_axis[1], th_y_axis[2], 
        color='green', length=arrow_length, label='Y-axis', linewidth=2)
ax.quiver(th_origin[0], th_origin[1], th_origin[2], 
        th_z_axis[0], th_z_axis[1], th_z_axis[2], 
        color='blue', length=arrow_length, label='Z-axis', linewidth=2)
plt.tight_layout()
plt.legend()
plt.show()