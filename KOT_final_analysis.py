import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from roboticstoolbox import DHRobot, RevoluteDH
import matplotlib.pyplot as plt
from spatialmath import SE3
from spatialmath.base import tr2rpy
from sklearn.metrics.pairwise import euclidean_distances 
from scipy.optimize import curve_fit
from hand_metrics import orientation_similarity_metric

def joint_trajectory(start_angle,end_angle,t_vec,T_dur):
    return np.array([start_angle + (end_angle - start_angle) * (
        10 * (t/T_dur)**3 - 15 * (t/T_dur)**4 + 6 * (t/T_dur)**5) for t in t_vec])

def compute_workspace(ET):
    workspace_pts = np.zeros((3,N_sample))
    workspace_pts_config = []
    for i in range(N_sample):
        if(ET.name=="Thumb"):
            finger_angles = np.array([[0],-np.radians(47)-np.random.choice(TH_CMC_ROM_a,1,replace=False),np.random.choice(TH_CMC_ROM_f,1,replace=False),np.random.choice(TH_MCP_ROM_a,1,replace=False),np.random.choice(TH_MCP_ROM_f,1,replace=False),np.random.choice(TH_IP_ROM,1,replace=False)])
        elif(ET.name=="LittleFinger"):
            finger_angles = np.array([[0],[0],np.random.choice(LF_CMC_ROM,1,replace=False),np.random.choice(LF_MCP_ROM_a,1,replace=False),np.random.choice(LF_MCP_ROM_f,1,replace=False),[0],np.random.choice(LF_PIP_ROM,1,replace=False),[0],np.random.choice(LF_DIP_ROM,1,replace=False)])     
        elif(ET.name=="LittleFingerProto2"):
            finger_angles = np.array([[0],[0],np.random.choice(LF_MCP_ROM_a,1,replace=False),np.random.choice(LF_MCP_ROM_f,1,replace=False),[0],np.random.choice(LF_PIP_ROM,1,replace=False),[0],np.random.choice(LF_DIP_ROM,1,replace=False)])     
        elif(ET.name=="IndexFinger"):
            finger_angles = np.array([[0],[0],np.random.choice(IF_MCP_ROM_a,1,replace=False),[0],np.random.choice(IF_MCP_ROM_f,1,replace=False),np.random.choice(IF_PIP_ROM,1,replace=False),np.random.choice(IF_DIP_ROM,1,replace=False)])        
        elif(ET.name=="MiddleFinger"):
            finger_angles = np.array([[0],np.random.choice(MF_MCP_ROM_a,1,replace=False),np.random.choice(MF_MCP_ROM_f,1,replace=False),np.random.choice(MF_PIP_ROM,1,replace=False),np.random.choice(MF_DIP_ROM,1,replace=False)])                      
        elif(ET.name=="RingFingerProto2"):
            finger_angles = np.array([[0],[0],np.random.choice(RF_MCP_ROM_a,1,replace=False),np.random.choice(RF_MCP_ROM_f,1,replace=False),[0],np.random.choice(RF_PIP_ROM,1,replace=False),[0],np.random.choice(RF_DIP_ROM,1,replace=False)])  
        else:
            finger_angles = np.array([[0],[0],np.random.choice(RF_CMC_ROM,1,replace=False),np.random.choice(RF_MCP_ROM_a,1,replace=False),np.random.choice(RF_MCP_ROM_f,1,replace=False),[0],np.random.choice(RF_PIP_ROM,1,replace=False),[0],np.random.choice(RF_DIP_ROM,1,replace=False)])  
        finger_angles = finger_angles.reshape(finger_angles.shape[0],)
        workspace_pts[:,i] = ET.fkine(finger_angles).A[0:3,3]
        workspace_pts_config.append(finger_angles)

    workspace_pts_config = np.array(workspace_pts_config).reshape((len(workspace_pts_config),len(workspace_pts_config[0])))
    return workspace_pts, workspace_pts_config

def compute_workspace_intersection_points(thumb_workspace, thumb_angles, opposing_finger_workspace, opposing_finger_angles):

    thumb_angles = np.array(thumb_angles)
    opposing_finger_angles = np.array(opposing_finger_angles)
    opposing_finger_workspace_points = []
    TH_workspace_points = []
    th_trial_config = []
    opposing_finger_trial_config = []

    for i in range(thumb_workspace.shape[1]):
        pt_point = thumb_workspace[:,i].reshape(1,3)
        # Compute pairwise euclidean distances between a point in the thumbs point cloud and the index finger's point cloud
        dist = euclidean_distances(opposing_finger_workspace.T,pt_point)
        j = np.argmin(dist,axis=0)[0]
        # If the distance is less than 5 mm then save the points and corresponding joint angles to an array and delete the saved points from the point clouds
        if(dist[j,0] < 0.001):
            opposing_finger_workspace_points.append(opposing_finger_workspace[:,j])
            TH_workspace_points.append(pt_point.flatten()) 
            th_trial_config.append(thumb_angles[i,:])
            opposing_finger_trial_config.append(opposing_finger_angles[j,:])
            opposing_finger_workspace = np.delete(opposing_finger_workspace,j,axis=1)
            opposing_finger_angles = np.delete(opposing_finger_angles,j,axis=0)

    opposing_finger_workspace_points = np.array(opposing_finger_workspace_points).T
    TH_workspace_points = np.array(TH_workspace_points).T
    th_trial_config = np.array(th_trial_config)
    opposing_finger_trial_config = np.array(opposing_finger_trial_config)

    return TH_workspace_points, th_trial_config, opposing_finger_workspace_points, opposing_finger_trial_config

N_sample = 5000  # Number of samples
np.random.seed(0)

TH = DHRobot([
    RevoluteDH(a=0.028, alpha=0, d=0.02, offset=np.radians(153.5)),
    RevoluteDH(a=0.0, alpha= np.pi/2, d=0.0),
    RevoluteDH(a=0.048945, alpha= -np.radians(150), d=0.0),
    RevoluteDH(a=0.0085, alpha=-np.pi/2, d=0.0),
    RevoluteDH(a=0.03822, alpha=0.0, d=0.0),
    RevoluteDH(a=0.03081, alpha=0.0, d=0.0),
], name='Thumb')

TH_CMC_ROM_a = np.linspace(-np.radians(15),np.radians(60),N_sample)
TH_CMC_ROM_f = np.linspace(0,np.radians(60),N_sample)
TH_MCP_ROM_a = np.linspace(0,np.radians(30),N_sample)
TH_MCP_ROM_f = np.linspace(0,np.radians(55),N_sample)
TH_IP_ROM = np.linspace(0,np.radians(80),N_sample)


IF = DHRobot([
    RevoluteDH(a=0.0, alpha=0, d=0, offset=np.radians(106.5)),
    RevoluteDH(a=0.088, alpha=0, d=0, offset=0.0), 
    RevoluteDH(a=0.0085, alpha=0.0, d=0),
    RevoluteDH(a=0.0, alpha= np.pi/2, d=0, offset=-np.radians(10)),
    RevoluteDH(a=0.047775, alpha= 0, d=0),
    RevoluteDH(a=0.027885, alpha= 0, d=0),
    RevoluteDH(a=0.018915, alpha=0, d=0),
], name='IndexFinger')

IF_MCP_ROM_a = np.linspace(-np.pi/6,np.pi/6,N_sample)
IF_MCP_ROM_f = np.linspace(0,(4*np.pi)/9,N_sample)
IF_PIP_ROM = np.linspace(0,(5*np.pi)/9,N_sample)
IF_DIP_ROM = np.linspace(0,np.pi/2,N_sample)

MF = DHRobot([
    RevoluteDH(a=0.087, alpha=0.0, d=0.0, offset=np.pi/2),
    RevoluteDH(a=0.0085, alpha = np.pi/2, d=0.0),
    RevoluteDH(a=0.05187, alpha=0.0, d=0.0),
    RevoluteDH(a=0.03315, alpha=0.0, d=0.0),
    RevoluteDH(a=0.018915, alpha=0.0, d=0.0),
], name='MiddleFinger')

MF_MCP_ROM_a = np.linspace(-np.radians(22.5),np.radians(22.5),N_sample)
MF_MCP_ROM_f = np.linspace(0,np.radians(80),N_sample)
MF_PIP_ROM = np.linspace(0,np.radians(100),N_sample)
MF_DIP_ROM = np.linspace(0,np.pi/2,N_sample)

RF = DHRobot([
    RevoluteDH(a=0.0, alpha=0.0, d=0.0, offset=np.radians(72.5)),
    RevoluteDH(a=0.031, alpha=np.pi/2, d=0.0, offset=0.0),
    RevoluteDH(a=0.0485, alpha=np.pi/2, d=0.0),
    RevoluteDH(a=0.0085, alpha= -np.pi/2, d=0.0),
    RevoluteDH(a=0.04758, alpha= -np.pi/2, d=0.0),
    RevoluteDH(a=0.0, alpha=np.pi/2, d=0.0, offset=np.radians(7)),
    RevoluteDH(a=0.032175, alpha=-np.pi/2, d=0.0),
    RevoluteDH(a=0.0, alpha=np.pi/2, d=0.0,offset=np.radians(7)),
    RevoluteDH(a=0.02865, alpha=0.0, d=0.0),
], name='RingFinger')

RF_proto2 = DHRobot([
    RevoluteDH(a=0.0, alpha=0.0, d=0.0, offset=np.radians(72.5)),
    RevoluteDH(a=0.0795, alpha=np.pi, d=0.0, offset=0.0),
    RevoluteDH(a=0.0085, alpha=-np.pi/2, d=0.0),
    RevoluteDH(a=0.04758, alpha= -np.pi/2, d=0.0),
    RevoluteDH(a=0.0, alpha=np.pi/2, d=0.0, offset=np.radians(7)),
    RevoluteDH(a=0.032175, alpha=-np.pi/2, d=0.0),
    RevoluteDH(a=0.0, alpha=np.pi/2, d=0.0,offset=np.radians(7)),
    RevoluteDH(a=0.020865, alpha=0.0, d=0.0),
], name='RingFingerProto2')

RF_CMC_ROM = np.linspace(0,np.radians(10),N_sample)
RF_MCP_ROM_a = np.linspace(-np.radians(22.5),np.radians(22.5),N_sample)
RF_MCP_ROM_f = np.linspace(0,np.radians(80),N_sample)
RF_PIP_ROM = np.linspace(0,np.radians(100),N_sample)
RF_DIP_ROM = np.linspace(0,np.pi/2,N_sample)

LF = DHRobot([
    RevoluteDH(a=0.0, alpha=0.0, d=0.0, offset=np.radians(63.5)),
    RevoluteDH(a=0.033, alpha=np.pi/2, d=0.0, offset=0.0), 
    RevoluteDH(a=0.043, alpha= np.pi/2, d=0.0),
    RevoluteDH(a=0.0085, alpha= -np.pi/2, d=0.0),
    RevoluteDH(a=0.03978, alpha= -np.pi/2, d=0.0),
    RevoluteDH(a=0.0, alpha=np.pi/2, d=0.0, offset=np.radians(10)),
    RevoluteDH(a=0.022815, alpha=-np.pi/2, d=0.0),
    RevoluteDH(a=0.0, alpha=np.pi/2, d=0.0, offset=np.radians(10)),
    RevoluteDH(a=0.018135, alpha=0.0, d=0.0),
], name='LittleFinger')

LF_proto2 = DHRobot([
    RevoluteDH(a=0.0, alpha=0.0, d=0.0, offset=np.radians(63.5)),
    RevoluteDH(a=0.076, alpha=np.pi, d=0.0, offset=0.0),
    RevoluteDH(a=0.0085, alpha=-np.pi/2, d=0.0),
    RevoluteDH(a=0.03978, alpha= -np.pi/2, d=0.0),
    RevoluteDH(a=0.0, alpha=np.pi/2, d=0.0, offset=np.radians(10)),
    RevoluteDH(a=0.022815, alpha=-np.pi/2, d=0.0),
    RevoluteDH(a=0.0, alpha=np.pi/2, d=0.0,offset=np.radians(10)),
    RevoluteDH(a=0.018135, alpha=0.0, d=0.0),
], name='LittleFingerProto2')

LF_CMC_ROM = np.linspace(0,np.radians(20),N_sample)
LF_MCP_ROM_a = np.linspace(-np.radians(25),np.radians(25),N_sample)
LF_MCP_ROM_f = np.linspace(0,np.radians(80),N_sample)
LF_PIP_ROM = np.linspace(0,np.radians(100),N_sample)
LF_DIP_ROM = np.linspace(0,np.pi/2,N_sample)

thumb_wksp, thumb_jt_angles = compute_workspace(TH)
proto2_thumb_wksp, proto2_thumb_jt_angles = compute_workspace(TH)
little_finger_wksp, little_finger_jt_angles = compute_workspace(LF)
little_finger_wksp_proto2, little_finger_jt_angles_proto2 = compute_workspace(LF_proto2)
thumb_intersection_points, thumb_intersection_angles, little_finger_intersection_points, little_finger_intersection_angles = compute_workspace_intersection_points(thumb_wksp,thumb_jt_angles,little_finger_wksp,little_finger_jt_angles)
thumb_intersection_points_proto2, thumb_intersection_angles_proto2, little_finger_intersection_points_proto2, little_finger_intersection_angles_proto2 = compute_workspace_intersection_points(proto2_thumb_wksp,proto2_thumb_jt_angles,little_finger_wksp_proto2,little_finger_jt_angles_proto2)
print(thumb_intersection_points.shape)
print(thumb_intersection_points_proto2.shape)
sim_metric_proto1 = np.zeros(little_finger_intersection_points.shape[1])
little_finger_mdof_orientation_euler_angles = np.zeros((little_finger_intersection_points.shape[1],3))

for i in range(thumb_intersection_points.shape[1]):
    # little_finger_mdof_orientation_euler_angles[i,:] = np.degrees(LF.fkine(little_finger_intersection_angles[i,:]).eul())
    Ot = TH.fkine(thumb_intersection_angles[i,:])
    Oi = LF.fkine(little_finger_intersection_angles[i,:])
    ideal_rot = (Oi.inv()@Ot).R
    little_finger_mdof_orientation_euler_angles[i,:] = tr2rpy(ideal_rot,unit="deg",order="zyx")
    eul_rot = SE3.Rz(np.radians(little_finger_mdof_orientation_euler_angles[i,2]))*SE3.Ry(np.radians(little_finger_mdof_orientation_euler_angles[i,1]))*SE3.Rx(np.radians(little_finger_mdof_orientation_euler_angles[i,0])).A[0:3,0:3]
    condition = np.allclose(ideal_rot, eul_rot)
    if not condition:
        print("Orientation Mismatch")
        raise AssertionError()
    Ot = Ot.A[0:3,0:3]
    Oi = Oi.A[0:3,0:3]
    sim_metric_proto1[i] = orientation_similarity_metric(Ot,Oi)

sim_metric_proto2 = np.zeros(little_finger_intersection_points_proto2.shape[1])
little_finger_proto2_orientation_euler_angles = np.zeros((little_finger_intersection_points_proto2.shape[1],3))

for i in range(little_finger_intersection_points_proto2.shape[1]):
    # little_finger_proto2_orientation_euler_angles[i,:] = np.degrees(LF_proto2.fkine(little_finger_intersection_angles_proto2[i,:]).eul())
    Ot = TH.fkine(thumb_intersection_angles_proto2[i,:])
    Oi = LF_proto2.fkine(little_finger_intersection_angles_proto2[i,:])
    ideal_rot = (Oi.inv()@Ot).R
    little_finger_proto2_orientation_euler_angles[i,:] = tr2rpy(ideal_rot,unit="deg",order="zyx")
    eul_rot = SE3.Rz(np.radians(little_finger_proto2_orientation_euler_angles[i,2]))*SE3.Ry(np.radians(little_finger_proto2_orientation_euler_angles[i,1]))*SE3.Rx(np.radians(little_finger_proto2_orientation_euler_angles[i,0])).A[0:3,0:3]
    condition = np.allclose(ideal_rot, eul_rot)
    if not condition:
        print("Orientation Mismatch")
        raise AssertionError()
    Ot = Ot.A[0:3,0:3]
    Oi = Oi.A[0:3,0:3]
    sim_metric_proto2[i] = orientation_similarity_metric(Ot,Oi)

j1 = np.argmax(sim_metric_proto1)
i1 = np.argmin(sim_metric_proto1)

z1 = np.argmax(little_finger_mdof_orientation_euler_angles,axis=0)
z2 = np.argmax(little_finger_proto2_orientation_euler_angles,axis=0)

j2 = np.argmax(sim_metric_proto2)
i2 = np.argmin(sim_metric_proto2)

th_ip = TH.fkine_all([0,-np.radians(47),0,0,0,0,0]).t
if_ip = IF.fkine_all([0,0,0,0,0,0,0]).t
mf_ip = MF.fkine_all([0,0,0,0,0]).t
rf_ip = RF.fkine_all([0,0,0,0,0,0,0,0,0]).t
lf_ip = LF.fkine_all([0,0,0,0,0,0,0,0,0]).t
lf2_ip = LF_proto2.fkine_all([0,0,0,0,0,0,0,0]).t

# fig2 = plt.figure()
# ax = plt.axes(projection='3d')
# ax.view_init(azim=-60, elev=30)
# ax.scatter(thumb_intersection_points[0,z1[0]],thumb_intersection_points[1,z1[0]],thumb_intersection_points[2,z1[0]],marker='^',color='blue',label='Max Euler Angle about X axis(Roll)')
# ax.scatter(thumb_intersection_points[0,z1[1]],thumb_intersection_points[1,z1[1]],thumb_intersection_points[2,z1[1]],marker='^',color='red',label='Max Euler Angle about Y axis(Pitch)')
# ax.scatter(thumb_intersection_points[0,z1[2]],thumb_intersection_points[1,z1[2]],thumb_intersection_points[2,z1[2]],marker='^',color='green',label='Max Euler Angle about Z axis(Yaw)')
# ax.plot(th_ip[:,0],th_ip[:,1],th_ip[:,2], c='g',marker='x',label="Thumb")
# ax.plot(lf_ip[:,0],lf_ip[:,1],lf_ip[:,2], c='orange',marker='x',label="LF")
# ax.plot(if_ip[:,0],if_ip[:,1],if_ip[:,2],c='blue',marker='x',label="IF")
# ax.plot(rf_ip[:,0],rf_ip[:,1],rf_ip[:,2],c='red',marker='x',label="RF")
# ax.plot(mf_ip[:,0],mf_ip[:,1],mf_ip[:,2], c='magenta',marker='x',label="MF")
# plt.legend()
# plt.tight_layout(pad=3.0)
# plt.title("Max of euler angles in workspace intersection for MAX DOF")
# plt.savefig("KOT optimal points",dpi=300)

# fig3 = plt.figure()
# ax = plt.axes(projection='3d')
# ax.view_init(azim=-60, elev=30)
# ax.scatter(thumb_intersection_points_proto2[0,z2[0]],thumb_intersection_points_proto2[1,z2[0]],thumb_intersection_points_proto2[2,z2[0]],marker='^',color='blue',label='Max Euler Angle about X axis(Roll)')
# ax.scatter(thumb_intersection_points_proto2[0,z2[1]],thumb_intersection_points_proto2[1,z2[1]],thumb_intersection_points_proto2[2,z2[1]],marker='^',color='red',label='Max Euler Angle about Y axis(Pitch)')
# ax.scatter(thumb_intersection_points_proto2[0,z2[2]],thumb_intersection_points_proto2[1,z2[2]],thumb_intersection_points_proto2[2,z2[2]],marker='^',color='green',label='Max Euler Angle about Z axis(Yaw)')
# ax.plot(th_ip[:,0],th_ip[:,1],th_ip[:,2], c='g',marker='x',label="Thumb")
# ax.plot(lf2_ip[:,0],lf2_ip[:,1],lf2_ip[:,2], c='orange',marker='x',label="LF")
# ax.plot(if_ip[:,0],if_ip[:,1],if_ip[:,2],c='blue',marker='x',label="IF")
# ax.plot(rf_ip[:,0],rf_ip[:,1],rf_ip[:,2],c='red',marker='x',label="RF")
# ax.plot(mf_ip[:,0],mf_ip[:,1],mf_ip[:,2], c='magenta',marker='x',label="MF")
# plt.legend()
# plt.tight_layout(pad=3.0)
# plt.title("Max of euler angles in workspace intersection for Prototype2")
# plt.savefig("KOT optimal points 2",dpi=300)

# fig4 = plt.figure()
# ax = plt.axes(projection='3d')
# ax.view_init(azim=-60, elev=30)
# ax.scatter(thumb_intersection_points[0,:],thumb_intersection_points[1,:],thumb_intersection_points[2,:],marker='^',color='blue',label='Thumb')
# ax.scatter(little_finger_intersection_points[0,:],little_finger_intersection_points[1,:],little_finger_intersection_points[2,:],marker='o',color='red',label='Little finger')
# ax.plot(th_ip[:,0],th_ip[:,1],th_ip[:,2], c='g',marker='x',label="Thumb")
# ax.plot(lf_ip[:,0],lf_ip[:,1],lf_ip[:,2], c='orange',marker='x',label="LF")
# ax.plot(if_ip[:,0],if_ip[:,1],if_ip[:,2],c='blue',marker='x',label="IF")
# ax.plot(rf_ip[:,0],rf_ip[:,1],rf_ip[:,2],c='red',marker='x',label="RF")
# ax.plot(mf_ip[:,0],mf_ip[:,1],mf_ip[:,2], c='magenta',marker='x',label="MF")
# plt.legend()
# plt.title("MAX DOF workspace intersection")
# plt.savefig("MAX DOF workspace intersection",dpi=300)

# fig5 = plt.figure()
# ax = plt.axes(projection='3d')
# ax.view_init(azim=-60, elev=30)
# ax.scatter(thumb_intersection_points_proto2[0,:],thumb_intersection_points_proto2[1,:],thumb_intersection_points_proto2[2,:],marker='^',color='blue',label='Thumb')
# ax.scatter(little_finger_intersection_points_proto2[0,:],little_finger_intersection_points_proto2[1,:],little_finger_intersection_points_proto2[2,:],marker='o',color='red',label='Little finger')
# ax.plot(th_ip[:,0],th_ip[:,1],th_ip[:,2], c='g',marker='x',label="Thumb")
# # ax.plot(lf_ip[:,0],lf_ip[:,1],lf_ip[:,2], c='orange',marker='x',label="LF")
# ax.plot(lf2_ip[:,0],lf2_ip[:,1],lf2_ip[:,2], c='orange',marker='x',label="LF")
# ax.plot(if_ip[:,0],if_ip[:,1],if_ip[:,2],c='blue',marker='x',label="IF")
# ax.plot(rf_ip[:,0],rf_ip[:,1],rf_ip[:,2],c='red',marker='x',label="RF")
# ax.plot(mf_ip[:,0],mf_ip[:,1],mf_ip[:,2], c='magenta',marker='x',label="MF")
# plt.legend()
# plt.title("Prototype 2 workspace intersection")
# plt.savefig("Prototype 2 workspace intersection",dpi=300)


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

fig2 = plt.figure()
plt.boxplot([sim_metric_proto1, sim_metric_proto2], notch=False, patch_artist=True, showmeans=True, showfliers=False,
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
plt.legend()
# plt.savefig('Orientation Mismatch KOT.png', dpi=300)

# fig3, ax = plt.subplots(1,3, figsize=(10,5))
# ax[0].boxplot([little_finger_mdof_orientation_euler_angles[:,2], little_finger_proto2_orientation_euler_angles[:,2]], notch=False, patch_artist=True, showfliers=False,
#             boxprops=dict(facecolor='lightblue', color='blue'),
#             whiskerprops=dict(color='black'),
#             capprops=dict(color='blue'),
#             medianprops=dict(color='red'))
# ax[0].axhline(y=180, color='k', linewidth=2, linestyle='--', xmin=xmin1, xmax=xmax1, label='Desired Euler Angle')
# ax[0].axhline(y=180, color='k', linewidth=2, linestyle='--', xmin=xmin2, xmax=xmax2)
# ax[0].set_xticks([1, 2], ['Max DOF', 'Prototype 2'])
# ax[0].set_ylabel('Angle (deg)')
# ax[0].set_title('Variation of Euler angle about Z axis (Yaw)')

# ax[1].boxplot([little_finger_mdof_orientation_euler_angles[:,1], little_finger_proto2_orientation_euler_angles[:,1]], notch=False, patch_artist=True, showfliers=False,
#             boxprops=dict(facecolor='lightblue', color='blue'),
#             whiskerprops=dict(color='black'),
#             capprops=dict(color='blue'),
#             medianprops=dict(color='red'))
# ax[1].axhline(y=0, color='k', linewidth=2, linestyle='--', xmin=xmin1, xmax=xmax1, label='Desired Euler Angle')
# ax[1].axhline(y=0, color='k', linewidth=2, linestyle='--', xmin=xmin2, xmax=xmax2)
# ax[1].set_xticks([1, 2], ['Max DOF', 'Prototype 2'])
# ax[1].set_ylabel('Angle (deg)')
# ax[1].set_title('Variation of Euler angle about Y axis (Pitch)')

# ax[2].boxplot([little_finger_mdof_orientation_euler_angles[:,0], little_finger_proto2_orientation_euler_angles[:,0]], notch=False, patch_artist=True, showfliers=True,  whis=180,
#             boxprops=dict(facecolor='lightblue', color='blue'),
#             whiskerprops=dict(color='black'),
#             capprops=dict(color='blue'),
#             medianprops=dict(color='red'))
# ax[2].axhline(y=120, color='k', linewidth=2, linestyle='--', xmin=xmin1, xmax=xmax1, label='Desired Euler Angle')
# ax[2].axhline(y=120, color='k', linewidth=2, linestyle='--', xmin=xmin2, xmax=xmax2)
# ax[2].set_xticks([1, 2], ['Max DOF', 'Prototype 2'])
# ax[2].set_ylabel('Angle (deg)')
# ax[2].set_title('Variation of Euler angle about X axis (Roll)')
# plt.tight_layout()
# plt.savefig('Euler angles from little finger to thumb.png', dpi=300)


# MAX DOF Hand
# th_end_effector_frame = TH.fkine(thumb_intersection_angles[j1,:]).A
# lf_end_effector_frame = LF.fkine(little_finger_intersection_angles[j1,:]).A

# # Extract transformed coordinates
# th_origin = th_end_effector_frame[0:3, 3]
# th_x_axis = th_end_effector_frame[0:3, 0]
# th_y_axis = th_end_effector_frame[0:3, 1]
# th_z_axis = th_end_effector_frame[0:3, 2]

# lf_origin = lf_end_effector_frame[0:3, 3]
# lf_x_axis = lf_end_effector_frame[0:3, 0]
# lf_y_axis = lf_end_effector_frame[0:3, 1]
# lf_z_axis = lf_end_effector_frame[0:3, 2]

# th_ip = TH.fkine_all(thumb_intersection_angles[j1,:]).t
# lf_ip = LF.fkine_all(little_finger_intersection_angles[j1,:]).t

# # lf_proto2_ip = LF_proto2.fkine_all(index_finger_intersection_angles_proto2[0,:]).t
# if_ip = IF.fkine_all([0,0,0,0,0,0,0]).t
# mf_ip = MF.fkine_all([0,0,0,0,0]).t
# rf_ip = RF.fkine_all([0,0,0,0,0,0,0,0,0]).t

# arrow_length = 0.02  # Adjust this value to shorten the arrows


# fig6 = plt.figure()
# ax = plt.axes(projection='3d')
# ax.quiver(th_origin[0], th_origin[1], th_origin[2], 
#         th_x_axis[0], th_x_axis[1], th_x_axis[2], 
#         color='black', length=arrow_length, label='TH X-axis', linewidth=2)
# ax.quiver(th_origin[0], th_origin[1], th_origin[2], 
#         th_y_axis[0], th_y_axis[1], th_y_axis[2], 
#         color='magenta', length=arrow_length, label='TH Y-axis', linewidth=2)
# ax.quiver(th_origin[0], th_origin[1], th_origin[2], 
#         th_z_axis[0], th_z_axis[1], th_z_axis[2], 
#         color='gold', length=arrow_length, label='TH Z-axis', linewidth=2)
# ax.quiver(lf_origin[0], lf_origin[1], lf_origin[2], 
#         lf_x_axis[0], lf_x_axis[1], lf_x_axis[2], 
#         color='r', length=arrow_length, label='LF X-axis', linewidth=2)
# ax.quiver(lf_origin[0], lf_origin[1], lf_origin[2], 
#         lf_y_axis[0], lf_y_axis[1], lf_y_axis[2], 
#         color='g', length=arrow_length, label='LF Y-axis', linewidth=2)
# ax.quiver(lf_origin[0], lf_origin[1], lf_origin[2], 
#         lf_z_axis[0], lf_z_axis[1], lf_z_axis[2], 
#         color='b', length=arrow_length, label='LF Z-axis', linewidth=2)
# ax.plot(th_ip[:,0],th_ip[:,1],th_ip[:,2], c='purple', marker='x', label="Thumb")
# # ax.plot(lf_proto2_ip[:,0],lf_proto2_ip[:,1],lf_proto2_ip[:,2], c='brown', marker='x', label="LF")
# ax.plot(lf_ip[:,0],lf_ip[:,1],lf_ip[:,2], c='olive', marker='x', label="LF")
# ax.plot(if_ip[:,0],if_ip[:,1],if_ip[:,2],c='blue',marker='x',label="IF")
# ax.plot(rf_ip[:,0],rf_ip[:,1],rf_ip[:,2],c='red',marker='x',label="RF")
# ax.plot(mf_ip[:,0],mf_ip[:,1],mf_ip[:,2],c='lime',marker='x',label="MF")
# plt.legend()
# plt.title("KOT pose of thumb and little finger for MAX DOF")
# plt.savefig("KOT pose MAX DOF", dpi=300)

# th_end_effector_frame = TH.fkine(thumb_intersection_angles[i1,:]).A
# lf_end_effector_frame = LF.fkine(little_finger_intersection_angles[i1,:]).A

# # Extract transformed coordinates
# th_origin = th_end_effector_frame[0:3, 3]
# th_x_axis = th_end_effector_frame[0:3, 0]
# th_y_axis = th_end_effector_frame[0:3, 1]
# th_z_axis = th_end_effector_frame[0:3, 2]

# lf_origin = lf_end_effector_frame[0:3, 3]
# lf_x_axis = lf_end_effector_frame[0:3, 0]
# lf_y_axis = lf_end_effector_frame[0:3, 1]
# lf_z_axis = lf_end_effector_frame[0:3, 2]

# th_ip = TH.fkine_all(thumb_intersection_angles[i1,:]).t
# lf_ip = LF.fkine_all(little_finger_intersection_angles[i1,:]).t

# # lf_proto2_ip = LF_proto2.fkine_all(index_finger_intersection_angles_proto2[0,:]).t
# if_ip = IF.fkine_all([0,0,0,0,0,0,0]).t
# mf_ip = MF.fkine_all([0,0,0,0,0]).t
# rf_ip = RF.fkine_all([0,0,0,0,0,0,0,0,0]).t

# fig7 = plt.figure()
# ax = plt.axes(projection='3d')
# ax.quiver(th_origin[0], th_origin[1], th_origin[2], 
#         th_x_axis[0], th_x_axis[1], th_x_axis[2], 
#         color='black', length=arrow_length, label='TH X-axis', linewidth=2)
# ax.quiver(th_origin[0], th_origin[1], th_origin[2], 
#         th_y_axis[0], th_y_axis[1], th_y_axis[2], 
#         color='magenta', length=arrow_length, label='TH Y-axis', linewidth=2)
# ax.quiver(th_origin[0], th_origin[1], th_origin[2], 
#         th_z_axis[0], th_z_axis[1], th_z_axis[2], 
#         color='gold', length=arrow_length, label='TH Z-axis', linewidth=2)
# ax.quiver(lf_origin[0], lf_origin[1], lf_origin[2], 
#         lf_x_axis[0], lf_x_axis[1], lf_x_axis[2], 
#         color='r', length=arrow_length, label='LF X-axis', linewidth=2)
# ax.quiver(lf_origin[0], lf_origin[1], lf_origin[2], 
#         lf_y_axis[0], lf_y_axis[1], lf_y_axis[2], 
#         color='g', length=arrow_length, label='LF Y-axis', linewidth=2)
# ax.quiver(lf_origin[0], lf_origin[1], lf_origin[2], 
#         lf_z_axis[0], lf_z_axis[1], lf_z_axis[2], 
#         color='b', length=arrow_length, label='LF Z-axis', linewidth=2)
# ax.plot(th_ip[:,0],th_ip[:,1],th_ip[:,2], c='purple', marker='x', label="Thumb")
# # ax.plot(lf_proto2_ip[:,0],lf_proto2_ip[:,1],lf_proto2_ip[:,2], c='brown', marker='x', label="LF")
# ax.plot(lf_ip[:,0],lf_ip[:,1],lf_ip[:,2], c='olive', marker='x', label="LF")
# ax.plot(if_ip[:,0],if_ip[:,1],if_ip[:,2],c='blue',marker='x',label="IF")
# ax.plot(rf_ip[:,0],rf_ip[:,1],rf_ip[:,2],c='red',marker='x',label="RF")
# ax.plot(mf_ip[:,0],mf_ip[:,1],mf_ip[:,2],c='lime',marker='x',label="MF")
# plt.legend()
# plt.title("KOT pose of thumb and little finger for MAX DOF")
# plt.savefig("KOT pose MAX DOF 2", dpi=300)

# Protoype 2

# th_end_effector_frame = TH.fkine(thumb_intersection_angles_proto2[j2,:]).A
# lf_end_effector_frame = LF_proto2.fkine(little_finger_intersection_angles_proto2[j2,:]).A

# # Extract transformed coordinates
# th_origin = th_end_effector_frame[0:3, 3]
# th_x_axis = th_end_effector_frame[0:3, 0]
# th_y_axis = th_end_effector_frame[0:3, 1]
# th_z_axis = th_end_effector_frame[0:3, 2]

# lf_origin = lf_end_effector_frame[0:3, 3]
# lf_x_axis = lf_end_effector_frame[0:3, 0]
# lf_y_axis = lf_end_effector_frame[0:3, 1]
# lf_z_axis = lf_end_effector_frame[0:3, 2]

# th_ip = TH.fkine_all(thumb_intersection_angles_proto2[j2,:]).t
# lf_proto2_ip = LF_proto2.fkine_all(little_finger_intersection_angles_proto2[j2,:]).t
# if_ip = IF.fkine_all([0,0,0,0,0,0,0]).t
# mf_ip = MF.fkine_all([0,0,0,0,0]).t
# rf_ip = RF.fkine_all([0,0,0,0,0,0,0,0,0]).t

arrow_length = 0.02  # Adjust this value to shorten the arrows


# fig8 = plt.figure()
# ax = plt.axes(projection='3d')
# ax.quiver(th_origin[0], th_origin[1], th_origin[2], 
#         th_x_axis[0], th_x_axis[1], th_x_axis[2], 
#         color='black', length=arrow_length, label='TH X-axis', linewidth=2)
# ax.quiver(th_origin[0], th_origin[1], th_origin[2], 
#         th_y_axis[0], th_y_axis[1], th_y_axis[2], 
#         color='magenta', length=arrow_length, label='TH Y-axis', linewidth=2)
# ax.quiver(th_origin[0], th_origin[1], th_origin[2], 
#         th_z_axis[0], th_z_axis[1], th_z_axis[2], 
#         color='gold', length=arrow_length, label='TH Z-axis', linewidth=2)
# ax.quiver(lf_origin[0], lf_origin[1], lf_origin[2], 
#         lf_x_axis[0], lf_x_axis[1], lf_x_axis[2], 
#         color='r', length=arrow_length, label='LF X-axis', linewidth=2)
# ax.quiver(lf_origin[0], lf_origin[1], lf_origin[2], 
#         lf_y_axis[0], lf_y_axis[1], lf_y_axis[2], 
#         color='g', length=arrow_length, label='LF Y-axis', linewidth=2)
# ax.quiver(lf_origin[0], lf_origin[1], lf_origin[2], 
#         lf_z_axis[0], lf_z_axis[1], lf_z_axis[2], 
#         color='b', length=arrow_length, label='LF Z-axis', linewidth=2)
# ax.plot(th_ip[:,0],th_ip[:,1],th_ip[:,2], c='purple', marker='x', label="Thumb")
# ax.plot(lf_proto2_ip[:,0],lf_proto2_ip[:,1],lf_proto2_ip[:,2], c='brown', marker='x', label="LF")
# # ax.plot(lf_ip[:,0],lf_ip[:,1],lf_ip[:,2], c='olive', marker='x', label="LF")
# ax.plot(if_ip[:,0],if_ip[:,1],if_ip[:,2],c='blue',marker='x',label="IF")
# ax.plot(rf_ip[:,0],rf_ip[:,1],rf_ip[:,2],c='red',marker='x',label="RF")
# ax.plot(mf_ip[:,0],mf_ip[:,1],mf_ip[:,2],c='lime',marker='x',label="MF")
# plt.legend()
# plt.title("KOT pose of thumb and little finger for Protoype 2")
# plt.savefig("KOT pose proto2", dpi=300)

# th_end_effector_frame = TH.fkine(thumb_intersection_angles_proto2[i2,:]).A
# lf_end_effector_frame = LF_proto2.fkine(little_finger_intersection_angles_proto2[i2,:]).A

# # Extract transformed coordinates
# th_origin = th_end_effector_frame[0:3, 3]
# th_x_axis = th_end_effector_frame[0:3, 0]
# th_y_axis = th_end_effector_frame[0:3, 1]
# th_z_axis = th_end_effector_frame[0:3, 2]

# lf_origin = lf_end_effector_frame[0:3, 3]
# lf_x_axis = lf_end_effector_frame[0:3, 0]
# lf_y_axis = lf_end_effector_frame[0:3, 1]
# lf_z_axis = lf_end_effector_frame[0:3, 2]

# th_ip = TH.fkine_all(thumb_intersection_angles_proto2[i2,:]).t
# lf_proto2_ip = LF_proto2.fkine_all(little_finger_intersection_angles_proto2[i2,:]).t
# if_ip = IF.fkine_all([0,0,0,0,0,0,0]).t
# mf_ip = MF.fkine_all([0,0,0,0,0]).t
# rf_ip = RF.fkine_all([0,0,0,0,0,0,0,0,0]).t

# fig9 = plt.figure()
# ax = plt.axes(projection='3d')
# ax.quiver(th_origin[0], th_origin[1], th_origin[2], 
#         th_x_axis[0], th_x_axis[1], th_x_axis[2], 
#         color='black', length=arrow_length, label='TH X-axis', linewidth=2)
# ax.quiver(th_origin[0], th_origin[1], th_origin[2], 
#         th_y_axis[0], th_y_axis[1], th_y_axis[2], 
#         color='magenta', length=arrow_length, label='TH Y-axis', linewidth=2)
# ax.quiver(th_origin[0], th_origin[1], th_origin[2], 
#         th_z_axis[0], th_z_axis[1], th_z_axis[2], 
#         color='gold', length=arrow_length, label='TH Z-axis', linewidth=2)
# ax.quiver(lf_origin[0], lf_origin[1], lf_origin[2], 
#         lf_x_axis[0], lf_x_axis[1], lf_x_axis[2], 
#         color='r', length=arrow_length, label='LF X-axis', linewidth=2)
# ax.quiver(lf_origin[0], lf_origin[1], lf_origin[2], 
#         lf_y_axis[0], lf_y_axis[1], lf_y_axis[2], 
#         color='g', length=arrow_length, label='LF Y-axis', linewidth=2)
# ax.quiver(lf_origin[0], lf_origin[1], lf_origin[2], 
#         lf_z_axis[0], lf_z_axis[1], lf_z_axis[2], 
#         color='b', length=arrow_length, label='LF Z-axis', linewidth=2)
# ax.plot(th_ip[:,0],th_ip[:,1],th_ip[:,2], c='purple', marker='x', label="Thumb")
# ax.plot(lf_proto2_ip[:,0],lf_proto2_ip[:,1],lf_proto2_ip[:,2], c='brown', marker='x', label="LF")
# # ax.plot(lf_ip[:,0],lf_ip[:,1],lf_ip[:,2], c='olive', marker='x', label="LF")
# ax.plot(if_ip[:,0],if_ip[:,1],if_ip[:,2],c='blue',marker='x',label="IF")
# ax.plot(rf_ip[:,0],rf_ip[:,1],rf_ip[:,2],c='red',marker='x',label="RF")
# ax.plot(mf_ip[:,0],mf_ip[:,1],mf_ip[:,2],c='lime',marker='x',label="MF")
# plt.legend()
# plt.title("KOT pose of thumb and little finger for Prototype 2")
# plt.savefig("KOT pose Proto2 2", dpi=300)

# Uncomment this code if you want to run the animation for the KOT for all the points in the workspace intersection

# T = 5  # in seconds
# time_steps = np.linspace(0, T, 100)  # Generate 100 time steps from 0 to T
# thumb_jshape = thumb_intersection_angles.shape
# little_jshape = little_finger_intersection_angles.shape
# t_shape = time_steps.shape
# thumb_trajectories = np.zeros((5,t_shape[0]))
# little_finger_trajectories = np.zeros((5,t_shape[0]))

# for j in range(sim_metric_proto1.shape[0]):
#     TH_theta_start = np.array([[-np.radians(47), thumb_intersection_angles[j,1]],[0, thumb_intersection_angles[j,2]],[0, thumb_intersection_angles[j,3]],[0, thumb_intersection_angles[j,4]],[0, thumb_intersection_angles[j,5]]])
#     LF_theta_start = np.array([[0, little_finger_intersection_angles[j,2]],[0, little_finger_intersection_angles[j,3]],[0, little_finger_intersection_angles[j,4]],[0, little_finger_intersection_angles[j,6]],[0, little_finger_intersection_angles[j,8]]])


#     Ot = TH.fkine(thumb_intersection_angles[j,:])
#     Oi = LF.fkine(little_finger_intersection_angles[j,:])

#     th_end_effector_frame = Ot.A
#     lf_end_effector_frame = Oi.A


#     th_origin = th_end_effector_frame[0:3, 3]
#     th_x_axis = th_end_effector_frame[0:3, 0]
#     # th_y_axis = th_end_effector_frame[0:3, 1]
#     # th_z_axis = th_end_effector_frame[0:3, 2]

#     # print(np.arccos(np.dot(th_x_axis,th_y_axis)/(np.linalg.norm(th_x_axis)*np.linalg.norm(th_y_axis))))
#     # print(np.arccos(np.dot(th_y_axis,th_z_axis)/(np.linalg.norm(th_y_axis)*np.linalg.norm(th_z_axis))))
#     # print(np.arccos(np.dot(th_x_axis,th_z_axis)/(np.linalg.norm(th_x_axis)*np.linalg.norm(th_z_axis))))

#     lf_origin = lf_end_effector_frame[0:3, 3]
#     lf_x_axis = lf_end_effector_frame[0:3, 0]
#     # lf_y_axis = lf_end_effector_frame[0:3, 1]
#     # lf_z_axis = lf_end_effector_frame[0:3, 2]

#     arrow_length = 0.02  

#     for i in range(TH_theta_start.shape[0]):
#         thumb_trajectories[i,:] = joint_trajectory(TH_theta_start[i,0],TH_theta_start[i,1],time_steps,T)

#     for i in range(LF_theta_start.shape[0]):
#         little_finger_trajectories[i,:] = joint_trajectory(LF_theta_start[i,0],LF_theta_start[i,1],time_steps,T)

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     # line1, = ax.plot([], [], [], c='g', linewidth=2, marker='x', label="Thumb")
#     # line2, = ax.plot([], [], [], c='r', linewidth=2, marker='x', label="Index Finger")
#     # line3, = ax.plot([], [], [], c='b', linewidth=2, marker='x', label="Middle Finger")
#     # line4, = ax.plot([], [], [], c='m', linewidth=2, marker='x', label="Ring Finger")
#     line5, = ax.plot([], [], [], c='orange', linewidth=2, marker='x', label="Little Finger")

#     ax.quiver(th_origin[0], th_origin[1], th_origin[2], 
#             th_x_axis[0], th_x_axis[1], th_x_axis[2], 
#             color='black', length=arrow_length, label='TH X-axis', linewidth=2)
#     # ax.quiver(th_origin[0], th_origin[1], th_origin[2], 
#     #         th_y_axis[0], th_y_axis[1], th_y_axis[2], 
#     #         color='magenta', length=arrow_length, label='TH Y-axis', linewidth=2)
#     # ax.quiver(th_origin[0], th_origin[1], th_origin[2], 
#     #         th_z_axis[0], th_z_axis[1], th_z_axis[2], 
#     #         color='gold', length=arrow_length, label='TH Z-axis', linewidth=2)
#     ax.quiver(lf_origin[0], lf_origin[1], lf_origin[2], 
#             lf_x_axis[0], lf_x_axis[1], lf_x_axis[2], 
#             color='r', length=arrow_length, label='LF X-axis', linewidth=2)
#     # ax.quiver(lf_origin[0], lf_origin[1], lf_origin[2], 
#     #         lf_y_axis[0], lf_y_axis[1], lf_y_axis[2], 
#     #         color='g', length=arrow_length, label='LF Y-axis', linewidth=2)
#     # ax.quiver(lf_origin[0], lf_origin[1], lf_origin[2], 
#     #         lf_z_axis[0], lf_z_axis[1], lf_z_axis[2], 
#     #         color='b', length=arrow_length, label='LF Z-axis', linewidth=2)


#     ax.set_xlim((-0.1,0.2))
#     ax.set_ylim((-0.1,0.2))
#     ax.set_zlim((0,0.07))

#     # ax.legend()


#     for i in range(t_shape[0]):
#         # qt = [0, thumb_trajectories[0,i], thumb_trajectories[1,i], thumb_trajectories[2,i], thumb_trajectories[3,i], thumb_trajectories[4,i]]
#         # qi = [0, 0, 0, 0, 0, 0, 0]
#         # qm = [0, 0, 0, 0, 0]
#         # qr = [0, 0, 0, 0, 0, 0, 0, 0, 0]
#         ql = [0, 0, little_finger_trajectories[0,i], little_finger_trajectories[1,i], little_finger_trajectories[2,i], 0, little_finger_trajectories[3,i], 0, little_finger_trajectories[4,i]]

#         # th_ip = TH.fkine_all(qt).t
#         # if_ip = IF.fkine_all(qi).t
#         # mf_ip = MF.fkine_all(qm).t
#         # rf_ip = RF.fkine_all(qr).t
#         lf_ip = LF.fkine_all(ql).t

#         # line1.set_data(th_ip[:, 0], th_ip[:, 1])
#         # line1.set_3d_properties(th_ip[:, 2])

#         # line2.set_data(if_ip[:, 0], if_ip[:, 1])
#         # line2.set_3d_properties(if_ip[:, 2])

#         # line3.set_data(mf_ip[:, 0], mf_ip[:, 1])
#         # line3.set_3d_properties(mf_ip[:, 2])

#         # line4.set_data(rf_ip[:, 0], rf_ip[:, 1])
#         # line4.set_3d_properties(rf_ip[:, 2])

#         line5.set_data(lf_ip[:, 0], lf_ip[:, 1])
#         line5.set_3d_properties(lf_ip[:, 2])

#         plt.legend()
#         plt.draw()
#         plt.pause(0.001)
#     plt.pause(10)
#     plt.close()
#     # plt.show()

plt.show()



