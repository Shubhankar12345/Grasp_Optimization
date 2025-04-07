import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH
import matplotlib.pyplot as plt
from spatialmath import SE3
from spatialmath.base import tr2rpy
from sklearn.metrics.pairwise import euclidean_distances 
from scipy.optimize import curve_fit
from hand_metrics import orientation_similarity_metric


np.random.seed(0)

class Handkinematics:
    def __init__(self, hand_info, N_sample=5000):
        """
        Creates an object for the hand prototype. Each finger in the hand is described by its Denavit-Hartenberg (DH) parameters, 
        an initial joint configuration, and a range of motion (ROM) for each joint.

        Args:
            hand_info (dict): A dictionary where each key is the name of a finger and the value is a tuple containing:
                - dh_params (list of tuples): Each tuple defines one joint in the finger and has the form:
                    (a, alpha, d, theta, isrevolute, isjointvariable)
                    where:
                        a (float): Link length in meters.
                        alpha (float): Link twist in radians.
                        d (float): Link offset in meters.
                        theta (float): Joint offset in radians.
                        isrevolute (bool): True if the joint is revolute.
                        isjointvariable (bool): True if the joint is variable.
                - initial_joint_configuration (list of float): The initial joint angles in radians for the finger.
                - finger_ROM (dict): Describes the range of motion for each joint in the finger kinematic chain. Each key 
                corresponds to a joint, and the value is a 1D array with `N_sample` points sampling the joint's range of motion.
            
            N_sample (int, optional): Number of points to sample the workspace. Defaults to 5000.

        Notes:
            - The DH parameters follow the standard Denavit-Hartenberg convention.
            - The `finger_ROM` allows for generating joint trajectories by providing sampled values within each joint's limits.

        Examples:
            hand_info = {'TH':(th_params, th_initial_config, TH_ROM),
                'IF': (if_params, if_inital_config, IF_ROM),
                'MF': (mf_params, mf_initial_config, MF_ROM),
                'RF': (rf_mdof_params, rf_cmc_config, RF_CMC_ROM),
                'LF': (lf_mdof_params, lf_cmc_config, LF_CMC_ROM)}

            th_params = [(0.028,0,0.02,np.radians(153.5),True,False),
            (0.0,np.pi/2,0.0,0.0,True,True),
            (0.048945,-np.radians(150),0.0,0.0,True,True),
            (0.0085,-np.pi/2,0.0,0.0,True,True),
            (0.03822,0.0,0.0,0.0,True,True),
            (0.03081,0.0,0.0,0.0,True,True)]

            th_initial_config = [-np.radians(47),0,0,0,0,0]

            TH_ROM = {'TH_CMC_a': -np.radians(47)-np.linspace(-np.radians(15),np.radians(60),N_sample),
            'TH_CMC_f': np.linspace(0,np.radians(60),N_sample),
            'TH_MCP_a': np.linspace(0,np.radians(30),N_sample),
            'TH_MCP_f': np.linspace(0,np.radians(55),N_sample),
            'TH_IP': np.linspace(0,np.radians(80),N_sample)}
            
            proto1 = {'TH':(th_params, th_initial_config, TH_ROM),
            'IF': (if_params, if_inital_config, IF_ROM),
            'MF': (mf_params, mf_initial_config, MF_ROM),
            'RF': (rf_mdof_params, rf_cmc_config, RF_CMC_ROM),
            'LF': (lf_mdof_params, lf_cmc_config, LF_CMC_ROM)}
            
            hand1 = Handkinematics(hand_info=proto1)
        """

        np.random.seed(0)
        self.hand_params = {}
        self.finger_ROM = {}
        self.sample_pts = N_sample
        self.hand_KC = {}
        self.hand_workspace = {}
        self.color = {}
        self.initial_config = {}
        self.initial_jt_config = {}
        self.jt = {}
        color = ['green','blue','magenta','red','orange']
        count = 0
        for key in hand_info.keys():
            DH_params = hand_info[key][0]
            jt_config = hand_info[key][1]
            n_frame = len(DH_params)
            DH_sz = len(DH_params[0])
            KC = DHRobot([RevoluteDH(a=DH_params[i][0],alpha=DH_params[i][1],d=DH_params[i][2],offset=DH_params[i][3])
                        if DH_params[i][DH_sz-2] else PrismaticDH(a=DH_params[i][0],alpha=DH_params[i][1],d=DH_params[i][2],offset=DH_params[i][3]) 
                        for i in range(n_frame)])
            self.hand_KC[key] = KC
            self.initial_config[key] = KC.fkine_all(jt_config).t
            self.initial_jt_config[key] = jt_config
            self.hand_params[key] = DH_params
            self.finger_ROM[key] = hand_info[key][2]
            self.color[key] = color[count]
            count += 1
        count = 0
        self.fingers = list(self.hand_params.keys())

    def compute_workspace(self):
        """
            Computes the workspace of each finger and then stores it in a dictionary 
            where each key is the name of a finger and value is a tuple (workspace_coords, workspace_jt_config)
            where
            workspace_coords (2D array of floats): 3d coordinates of end-effector of shape (3, N_sample)
            workspace_pts_config (2D array of floats): Joint angles for each pose of shape (N_sample, n_frames)
            where 
            n_frames is the number of frames in the kinematic chain of each finger
        """
        for key in self.hand_KC.keys():
            workspace_pts = np.zeros((3,self.sample_pts))
            workspace_pts_config = []
            n_frame = self.hand_KC[key].n
            robot_ROM = self.finger_ROM[key]
            rob_jts = list(robot_ROM.keys())
            for i in range(self.sample_pts):
                count = 0
                finger_angles = []
                for j in range(n_frame):
                    if(self.hand_params[key][j][-1]):
                        finger_angles.append(np.random.choice(robot_ROM[rob_jts[count]],replace=False))
                        count += 1
                    else:
                        finger_angles.append(0.0)
                workspace_pts[:,i] = self.hand_KC[key].fkine(finger_angles).A[0:3,3]
                workspace_pts_config.append(finger_angles)
            workspace_pts_config = np.array(workspace_pts_config).reshape((len(workspace_pts_config),len(workspace_pts_config[0])))
            self.hand_workspace[key] = (workspace_pts,workspace_pts_config)
        
    
    def plot_finger_workspace(self, ax, finger):
        """
        Plots a finger's workspace
        Args:
            ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axis object where the hand configuration 
            will be plotted. Typically obtained from `matplotlib` via `fig.add_subplot(111, projection='3d')
            finger (string): Key of the finger whose workspace you want to compute
        
        Examples:
            hand1.plot_finger_workspace(finger='TH')

        """

        c1 = self.color[finger]
        # fig1 = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.view_init(azim=-60, elev=30)
        ax.scatter(self.hand_workspace[finger][0][0,:],self.hand_workspace[finger][0][1,:],self.hand_workspace[finger][0][2,:],marker='^',color=c1,label=finger)
        self.plot_hand(ax=ax)
        plt.legend()
        plt.tight_layout(pad=3.0)
        plt.title(str(finger) + " " + "Workspace")
        return ax

    def compute_workspace_intersection_points(self, thumb, opposing_finger):
        """
        Compute the points lying in the intersection of the thumb workspace and opposing finger workspace.
        Args:
            thumb (string): Key of the thumb. Use the same key name used while initiating the instance of this class.
            opposing_finger (string): Key of the opposing finger. Use the same key name used while initiating the instance of this class.

        Returns:
            TH_workspace_points (2D array of floats): Return the thumb workspace points of shape (3, N_intersection) lying in the workspace intersection.
            th_trial_config (2D array of floats): Return the thumb joint configuration corresponding to the intersection points of shape (N_intersection, n_frames).
            opposing_finger_workspace_points (2D array of floats): Return the opposing finger workspace points of shape (3, N_intersection) lying in the workspace intersection.
            opposing_finger_trial_config (2D array of floats): Return the opposing finger joint configuration corresponding to the intersection points of shape (N_intersection, n_frames).
            where,
            N_intersection: no. of points in the workspace intersection.
            n_frames: no. of coordinate frames in finger kinematic chain.

        Examples:
            th_pts, th_jt_config, opp_pts, opp_jt_config = hand1.compute_workspace_intersection_points(thumb='TH',opposing_finger='LF')
        """

        thumb_angles = np.array(self.hand_workspace[thumb][1])
        thumb_workspace = self.hand_workspace[thumb][0]
        opposing_finger_angles = np.array(self.hand_workspace[opposing_finger][1])
        opposing_finger_workspace = self.hand_workspace[opposing_finger][0]
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
            if(dist[j,0] < 0.005):
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

    def KOTcomputation(self, th_angles, oppfinger_angles, oppfinger):
        """
        
        Computes the similarity between the orientation of thumb end-effector and opposing finger end-effector frame using a
        KOT inspired similarity metric. Also computes the euler angles for the transformation matrix of thumb with respect to little finger.

        Args:
            th_angles (2D array of floats): thumb joint configuration of shape (N_intersection, n_frames) corresponding to the thumb workspace points lying in the interesection of workspaces.
            oppfinger_angles (2D array of floats): opposing finger joint configuration of shape (N_intersection, n_frames) corresponding to the opposing finger workspace points lying in the interesection of workspaces.
            oppfinger (string): Key of the opposing finger. Use the same key name used while initiating the instance of this class.

        Raises:
            AssertionError: Checks whether the euler angle decomposition of the orientation is correct.

        Returns:
            sim_metric (1D array of floats): Values of similarity metric for each point in the workspace intersection.
            orientation_euler_angles (1D array of floats): Values of the euler angles for transformation thumb with respect to little finger.
        
        Examples:
            th_mdof, th_mdof_config, opp_mdof, opp_mdof_config = hand1.compute_workspace_intersection_points(thumb='TH',opposing_finger='LF')
            mdof_orientn_sim, mdof_orientn_euler = hand1.KOTcomputation(th_angles=th_mdof_config,oppfinger_angles=opp_mdof_config,oppfinger='LF')
        """
        sz = th_angles.shape
        sim_metric = np.zeros(sz[0])
        orientation_euler_angles = np.zeros((sz[0],3))

        for i in range(sz[0]):

            Ot = self.hand_KC['TH'].fkine(th_angles[i,:])
            Oi = self.hand_KC[oppfinger].fkine(oppfinger_angles[i,:])
            
            ideal_rot = (Oi.inv()@Ot).R
            orientation_euler_angles[i,:] = tr2rpy(ideal_rot,unit="deg",order="zyx")
            eul_rot = SE3.Rz(np.radians(orientation_euler_angles[i,2]))*SE3.Ry(np.radians(orientation_euler_angles[i,1]))*SE3.Rx(np.radians(orientation_euler_angles[i,0])).A[0:3,0:3]
            condition = np.allclose(ideal_rot, eul_rot)
            if not condition:
                print("Orientation Mismatch")
                raise AssertionError()
            Ot = Ot.A[0:3,0:3]
            Oi = Oi.A[0:3,0:3]
            sim_metric[i] = orientation_similarity_metric(Ot,Oi)
        
        return sim_metric, orientation_euler_angles
            
    def plot_wksp_intersection(self, th_wksp, oppfinger_wksp, config):
        """
        Plots the workspace intersection between thumb and opposing finger.

        Args:
            th_wksp (2D array of floats): thumb workspace.
            oppfinger_wksp (2D array of floats): opposing finger workspace.
            config (string): Name of the prototype 
        
        Examples:
            hand1.plot_wksp_intersection(th_wksp=th_mdof,oppfinger_wksp=opp_mdof,config='MAX DOF')
        """

        plt.figure()     
        ax = plt.axes(projection='3d')
        ax.view_init(azim=-60, elev=30)
        ax.scatter(th_wksp[0,:],th_wksp[1,:],th_wksp[2,:],marker='^',color='blue',label='Thumb')
        ax.scatter(oppfinger_wksp[0,:],oppfinger_wksp[1,:],oppfinger_wksp[2,:],marker='o',color='red',label='Little finger')
        self.plot_hand(ax=ax)
        plt.legend()
        plt.title(config + " hand configuration")

    @staticmethod
    def joint_trajectory(start_angle,end_angle,t_vec,T_dur):
        """
        Computes a minimum jerk trajectory for the kinematic animation of fingers while perform the KOT pose.
        Args:
            start_angle (float): initial condition for joint angle.
            end_angle (float): final condition for joint angle.
            t_vec (1D array of floats): vector of time steps.
            T_dur (float): duration of the trajectory.

        Returns:
            (1D array of floats): returns the trajectory
        """
        return np.array([start_angle + (end_angle - start_angle) * (
        10 * (t/T_dur)**3 - 15 * (t/T_dur)**4 + 6 * (t/T_dur)**5) for t in t_vec])

    @staticmethod
    def project_points_to_axis(points, axis, axis_point):
        """

        Project vector from cylinder center to the cylinder surface points onto the cylinder axis.

        Args:
            points (2D array of floats): cylinder surface points.
            axis (1D array of floats): vector describing the cylinder axis.
            axis_point (1D array of floats): vector describing the cylinder center.

        Returns:
            projected_points (2D array of floats): cylinder surface points projected onto the cylinder axis
        """
        
        points_centered = points - axis_point
        projections = np.dot(points_centered, axis)[:, None] * axis
        projected_points = projections + axis_point
        return projected_points, np.dot(points_centered, axis)

    def animate_hand(self, ax, th_jt_config, opp_jt_config, oppfinger):

        """
        Animation of the KOT pose

        Args:
            ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axis object where the hand configuration 
            will be plotted. Typically obtained from `matplotlib` via `fig.add_subplot(111, projection='3d')
            th_jt_config (1D array of floats): Thumb joint configuration corresponding to thumb position for a KOT pose.
            opp_jt_config (1D array of floats): Opposing finger joint configuration corresponding to the opposing finger workspace points lying in the interesection of workspaces.
            oppfinger (string): Key of the opposing finger. Use the same key name used while initiating the instance of this class.
        
        Examples:
            th_mdof, th_mdof_config, opp_mdof, opp_mdof_config = hand1.compute_workspace_intersection_points(thumb='TH',opposing_finger='LF')
            index1 = np.random.choice(th_mdof_config.shape[0])
            th_anim_config = th_mdof_config[index1,:]
            opp_anim_config = opp_mdof_config[index1,:]
            hand1.animate_hand(th_jt_config=th_anim_config,opp_jt_config=opp_anim_config,oppfinger='LF')    
        """
        T = 5  # in seconds
        time_steps = np.linspace(0, T, 1000)  # Generate 100 time steps from 0 to T
        t_shape = time_steps.shape
        TH_theta_start = []
        opp_theta_start = []
        n_thumb = len(self.initial_jt_config['TH'])
        n_opp = len(self.initial_jt_config[oppfinger])

        for j in range(n_thumb):
            if(self.hand_params['TH'][j][-1]):
                TH_theta_start.append((self.initial_jt_config['TH'][j],th_jt_config[j]))
            else:
                TH_theta_start.append((0.0,0.0))
        
        for i in range(n_opp):
            if(self.hand_params[oppfinger][i][-1]):
                opp_theta_start.append((self.initial_jt_config[oppfinger][i],opp_jt_config[i]))
            else:
                opp_theta_start.append((0.0,0.0))
        
        thumb_trajectories = np.zeros((n_thumb, t_shape[0]))
        opp_finger_trajectories = np.zeros((n_opp, t_shape[0]))

        for i in range(n_thumb):
            thumb_trajectories[i,:] = Handkinematics.joint_trajectory(TH_theta_start[i][0],TH_theta_start[i][1],time_steps,T)
        
        for i in range(n_opp):
            opp_finger_trajectories[i,:] = Handkinematics.joint_trajectory(opp_theta_start[i][0],opp_theta_start[i][1],time_steps,T)
        
        th_end_effector_frame = self.hand_KC['TH'].fkine(th_jt_config).A

        lf_end_effector_frame = self.hand_KC[oppfinger].fkine(opp_jt_config).A

        # Extract transformed coordinates
        th_origin = th_end_effector_frame[0:3, 3]
        th_x_axis = th_end_effector_frame[0:3, 0]
        th_y_axis = th_end_effector_frame[0:3, 1]
        th_z_axis = th_end_effector_frame[0:3, 2]

        lf_origin = lf_end_effector_frame[0:3, 3]
        lf_x_axis = lf_end_effector_frame[0:3, 0]
        lf_y_axis = lf_end_effector_frame[0:3, 1]
        lf_z_axis = lf_end_effector_frame[0:3, 2]

        arrow_length = 0.02  # Adjust this value to shorten the arrows

        # plt.figure()
        # ax = plt.axes(projection='3d')
        ax.set_xlim((-0.1,0.2))
        ax.set_ylim((-0.1,0.2))
        ax.set_zlim((0,0.07))
        line1, = ax.plot([], [], [], c='g', linewidth=2, marker='x', label="Thumb")
        line2, = ax.plot([], [], [], c='r', linewidth=2, marker='x', label="Index Finger")
        line3, = ax.plot([], [], [], c='b', linewidth=2, marker='x', label="Middle Finger")
        line4, = ax.plot([], [], [], c='m', linewidth=2, marker='x', label="Ring Finger")
        line5, = ax.plot([], [], [], c='orange', linewidth=2, marker='x', label="Little Finger")

        ax.quiver(th_origin[0], th_origin[1], th_origin[2], 
                th_x_axis[0], th_x_axis[1], th_x_axis[2], 
                color='black', length=arrow_length, label='TH X-axis', linewidth=1)
        ax.quiver(th_origin[0], th_origin[1], th_origin[2], 
                th_y_axis[0], th_y_axis[1], th_y_axis[2], 
                color='magenta', length=arrow_length, label='TH Y-axis', linewidth=1)
        ax.quiver(th_origin[0], th_origin[1], th_origin[2], 
                th_z_axis[0], th_z_axis[1], th_z_axis[2], 
                color='gold', length=arrow_length, label='TH Z-axis', linewidth=1)
        ax.quiver(lf_origin[0], lf_origin[1], lf_origin[2], 
                lf_x_axis[0], lf_x_axis[1], lf_x_axis[2], 
                color='r', length=arrow_length, label='LF X-axis', linewidth=1)
        ax.quiver(lf_origin[0], lf_origin[1], lf_origin[2], 
                lf_y_axis[0], lf_y_axis[1], lf_y_axis[2], 
                color='g', length=arrow_length, label='LF Y-axis', linewidth=1)
        ax.quiver(lf_origin[0], lf_origin[1], lf_origin[2], 
                lf_z_axis[0], lf_z_axis[1], lf_z_axis[2], 
                color='b', length=arrow_length, label='LF Z-axis', linewidth=1)
        
        for i in range(t_shape[0]):
            qt = thumb_trajectories[:,i]
            qi = self.initial_jt_config['IF']
            qm = self.initial_jt_config['MF']
            qr = self.initial_jt_config['RF']
            ql = self.initial_jt_config['LF']

            if(oppfinger=='LF'):
                ql = opp_finger_trajectories[:,i]
            elif(oppfinger=='RF'):
                rf_ip = opp_finger_trajectories[:,i]
            elif(oppfinger=='MF'):
                mf_ip = opp_finger_trajectories[:,i]
            else:
                if_ip = opp_finger_trajectories[:,i]

            th_ip = self.hand_KC['TH'].fkine_all(qt).t
            if_ip = self.hand_KC['IF'].fkine_all(qi).t
            mf_ip = self.hand_KC['MF'].fkine_all(qm).t
            rf_ip = self.hand_KC['RF'].fkine_all(qr).t
            lf_ip = self.hand_KC['LF'].fkine_all(ql).t
            line1.set_data(th_ip[:, 0], th_ip[:, 1])
            line1.set_3d_properties(th_ip[:, 2])

            line2.set_data(if_ip[:, 0], if_ip[:, 1])
            line2.set_3d_properties(if_ip[:, 2])

            line3.set_data(mf_ip[:, 0], mf_ip[:, 1])
            line3.set_3d_properties(mf_ip[:, 2])

            line4.set_data(rf_ip[:, 0], rf_ip[:, 1])
            line4.set_3d_properties(rf_ip[:, 2])

            line5.set_data(lf_ip[:, 0], lf_ip[:, 1])
            line5.set_3d_properties(lf_ip[:, 2])

            plt.draw()
            plt.pause(0.001)
        plt.legend()
        plt.title("KOT pose of thumb and little finger")
        
    @staticmethod    
    def project_finger_phalanxes(point, cylinder_axis, cylinder_center, radius, cylinder_lower_limit, cylinder_upper_limit):

        r_vec = point - cylinder_center # Vector from cylinder_center to the surface point

        r_vec_proj = np.dot(r_vec,cylinder_axis)

        # if(r_vec_proj<cylinder_lower_limit or r_vec_proj>cylinder_upper_limit):
        #     return False
        
        radial_dist = r_vec - r_vec_proj*cylinder_axis

        is_pt_inside_cylinder = np.linalg.norm(radial_dist) <= radius*0.9

        return is_pt_inside_cylinder

    @staticmethod
    def finger_phalanxes_intersect_cylinder(jt1, jt2, cylinder_axis, cylinder_center, radius, cylinder_lower_limit, cylinder_upper_limit, n_samples=30):
        
        for alpha in np.linspace(0,1,n_samples):
            pt = (1-alpha)*jt1 + alpha*jt2
            if(Handkinematics.project_finger_phalanxes(point=pt,cylinder_axis=cylinder_axis,cylinder_center=cylinder_center,
                radius=radius,cylinder_lower_limit=cylinder_lower_limit,cylinder_upper_limit=cylinder_upper_limit)):
                return True
        return False

    def filter_points_near_cylinder(self, finger, cylinder_points, cylinder_axis, cylinder_upward_axis, cylinder_radius, cylinder_lowerlimit, cylinder_upper_limit, tolerance=10):

        """
        Filter points that are outside the cylinder and within a given tolerance from the cylinder's surface, and also within the 
        vertical bounds of the cylinder from -0.1 to 0.1.
        
        Args:
            finger (string): Key of the finger for which you want to find the workspace points close to the intersection.
            cylinder_points (2D array of floats): Cylinder surface points of shape (3,N).
            cylinder_axis (1D array of floats): vector describing the cylinder axis.
            cylinder_radius (float): radius of cylinder.
            cylinder_lowerlimit (float): Axial lower limit for finding points close to the cylinder surface.
            cylinder_upper_limit (float): Axial upper limit for finding points close to the cylinder surface.
            tolerance (float) = Tolerance in the radial direction for finding points close to the cylinder surface. Defaults to 10 mm.

        Returns:
            nearby_points (2D array of floats): Points nearby the cylinder's surface of shape (N_f,3).
            nearby_points_angles (2D array of floats): Finger joint angles for points nearby the cylinder's surface of shape (N_f, n_frames).
            where
            N_f: Number of points close to the cylinder surface for a given finger.
            n_frames: number of coordinate frames in that finger.

        Examples:
            h1_th_candidate_pts = hand1.filter_points_near_cylinder(finger='TH',cylinder_points=transformed_pts.T,cylinder_axis=rot_matrix[:,2],
                                cylinder_radius=r,cylinder_lowerlimit=np.min(z),cylinder_upper_limit=np.min(z)/1.4)
        
        """
        workspace_points = self.hand_workspace[finger][0].T
        workspace_angles = self.hand_workspace[finger][1]

        self.r = cylinder_radius
        # Normalize the cylinder axis vector
        cylinder_axis = cylinder_axis / np.linalg.norm(cylinder_axis)
        self.cylinder_axis = cylinder_axis
        # Calculate the axis point (base point of the cylinder)
        axis_point = np.mean(cylinder_points, axis=0)  # Typically, adjust this if needed to match the -0.1 base point
        
        # Project workspace points onto the cylinder's axis
        projected_points, projections = Handkinematics.project_points_to_axis(workspace_points, cylinder_axis, axis_point)

        # Radial vectors from cylinder axis to workspace points
        radial_vectors = workspace_points - projected_points
        
        # Calculate perpendicular distances from workspace points to the axis
        distances_to_axis = np.linalg.norm(radial_vectors, axis=1)
        
        # Define the lower and upper limits for the radial distance
        min_distance = cylinder_radius  # Points must be outside the cylinder
        max_distance = cylinder_radius + tolerance / 1000.0  # Convert tolerance from mm to meters if necessary
        
        # Check axial bounds: projections must be within -0.1 to 0.1
        axial_bounds_mask = (projections >= cylinder_lowerlimit) & (projections <= cylinder_upper_limit)
        
        # Filter points that are outside the cylinder but within the tolerance from its surface
        radial_mask = (distances_to_axis > min_distance) & (distances_to_axis <= max_distance)

        # Mask for points on the upper half of the cylinder
        upper_surface_mask = (radial_vectors @ cylinder_upward_axis) > 0
        
        # Combine masks
        mask = axial_bounds_mask & radial_mask & upper_surface_mask
        candidate_nearby_points = workspace_points[mask]
        candidate_nearby_points_angles = workspace_angles[mask]
        cidx = candidate_nearby_points.shape[0]
        valid_idx = []
        
    
        for i in range(cidx):
            finger_jt_pos = self.hand_KC[finger].fkine_all(candidate_nearby_points_angles[i,:]).t
            n_jts = finger_jt_pos.shape[0]
            idx_flag = True
            for j in range(n_jts-1):
                # if(np.allclose(a=finger_jt_pos[j,:],b=finger_jt_pos[j+1,:])):
                #     continue
                # else:
                    # is_intersect = Handkinematics.segment_cylinder_intersection(
                    #             jt1=finger_jt_pos[j, :], jt2=finger_jt_pos[j+1, :],
                    #             cylinder_axis=cylinder_axis, cylinder_center=axis_point,
                    #             radius=cylinder_radius, lower_limit=cylinder_lowerlimit,
                    #             upper_limit=cylinder_upper_limit
                    #         )
                is_intersect = Handkinematics.finger_phalanxes_intersect_cylinder(
                            jt1=finger_jt_pos[j, :], jt2=finger_jt_pos[j+1, :],
                            cylinder_axis=cylinder_axis, cylinder_center=axis_point,
                            radius=cylinder_radius, cylinder_lower_limit=cylinder_lowerlimit,
                            cylinder_upper_limit=cylinder_upper_limit
                        )
                    # is_intersect2 = Handkinematics.project_finger_phalanxes(point=finger_jt_pos[j,:], 
                    #             cylinder_axis=cylinder_axis, cylinder_center=axis_point,
                    #             radius=cylinder_radius, cylinder_lower_limit=cylinder_lowerlimit,
                    #             cylinder_upper_limit=cylinder_upper_limit
                    #         )
                    # is_intersect3 = Handkinematics.project_finger_phalanxes(point=finger_jt_pos[j+1,:], 
                    #             cylinder_axis=cylinder_axis, cylinder_center=axis_point,
                    #             radius=cylinder_radius, cylinder_lower_limit=cylinder_lowerlimit,
                    #             cylinder_upper_limit=cylinder_upper_limit
                    #         )
                    
                if(is_intersect):
                    idx_flag = False
                    break

            if(idx_flag):
                valid_idx.append(i)

        
        nearby_points = candidate_nearby_points[valid_idx]
        nearby_points_angles = candidate_nearby_points_angles[valid_idx]
        # nearby_points = candidate_nearby_points
        # nearby_points_angles = candidate_nearby_points_angles


        return nearby_points, nearby_points_angles

    def plot_wksp_pts_close_to_cylinder(self, close_pts, cylinder_pts, config):

        """

        Plots the points close to the cylinder surface.

        Args:
            close_pts (list of 2D arrays of floats): List of points nearby the cylinder for each workspace
            cylinder_points (2D array of floats): Cylinder surface points of shape (3,N).
            config (string): Name of the prototype.
        
        Examples:
            h1_close_pts = (h1_th_candidate_pts[0],h1_if_candidate_pts[0],h1_mf_candidate_pts[0],h1_rf_candidate_pts[0],h1_lf_candidate_pts[0])
            hand1.plot_wksp_pts_close_to_cylinder(close_pts=h1_close_pts, cylinder_pts=transformed_pts,config='MAX DOF')
        """
        self.close_pts = close_pts
        self.cylinder_pts = cylinder_pts
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(close_pts[0][:,0],close_pts[0][:,1],close_pts[0][:,2],marker='^',color='green',label="TH")
        ax.scatter(close_pts[1][:,0],close_pts[1][:,1],close_pts[1][:,2],marker='^',color='blue',label="IF")
        ax.scatter(close_pts[2][:,0],close_pts[2][:,1],close_pts[2][:,2],marker='^',color='magenta',label="MF")
        ax.scatter(close_pts[3][:,0],close_pts[3][:,1],close_pts[3][:,2],marker='^',color='red',label="RF")
        ax.scatter(close_pts[4][:,0],close_pts[4][:,1],close_pts[4][:,2],marker='^',color='orange',label="LF")
        ax.scatter(cylinder_pts[0,:],cylinder_pts[1,:],cylinder_pts[2,:],marker='o',color='cyan',label='Cylinder')
        self.plot_hand(ax=ax)
        plt.legend()
        plt.tight_layout(pad=3.0)
        plt.title("Points close to cylinder surface for " + config)

    def best_grasp_location(self, finger, wksp_pts, wksp_angles, origin):

        finger_x_uvec = np.array(self.hand_KC[finger].fkine(wksp_angles).A)[:,0:3,0].T
        finger_F = finger_x_uvec
        origin = np.array([0.03,0.052,self.r])
        weight_cylinder = 1
        mu = 0.1
        Fr = np.zeros(wksp_pts.shape[0])
        Fa = np.zeros(wksp_pts.shape[0])
        Fnet = np.zeros(wksp_pts.shape[0])
        F_vec = np.zeros((3,wksp_pts.shape[0]))

        for i in range(wksp_pts.shape[0]):
            F_r, F_a, _ = Handkinematics.decompose_force(finger_F[:,i],wksp_pts[i,:],origin,self.cylinder_axis)
            F_net = F_a + mu*F_r - weight_cylinder
            Fr[i] = F_r
            Fa[i] = F_a
            Fnet[i] = F_net
            F_vec[:,i] = np.array([F_a,F_r,F_net])
        
        g_index = np.argmax(F_vec,axis=1)
        best_axial = (wksp_pts[g_index[0],:], self.hand_KC[finger].fkine_all(wksp_angles[g_index[0],:]).t, wksp_angles[g_index[0],:])
        best_radial = (wksp_pts[g_index[1],:], self.hand_KC[finger].fkine_all(wksp_angles[g_index[1],:]).t, wksp_angles[g_index[1],:])
        best_net = (wksp_pts[g_index[2],:], self.hand_KC[finger].fkine_all(wksp_angles[g_index[2],:]).t, wksp_angles[g_index[2],:])

        return best_axial, best_radial, best_net, Fa, Fr, Fnet

    def plot_best_grasp(self, finger, config, axial, radial, net):
        fig1 = plt.figure(figsize=(12, 6))
        ax1 = fig1.add_subplot(1,2,1,projection='3d')
        ax1.view_init(azim=-90, elev=90)
        ax1.scatter(axial[0],axial[1],axial[2],marker='^',color='blue',label='Max Axial')
        ax1.scatter(radial[0],radial[1],radial[2],marker='o',color='red',label='Max Radial')
        ax1.scatter(net[0],net[1],net[2],marker='s',color='black',label='Max Net')
        ax1.scatter(self.cylinder_pts[0,:],self.cylinder_pts[1,:],self.cylinder_pts[2,:],marker='o',color='cyan',label='Cylinder')
        self.plot_hand(ax=ax1)
        plt.legend()
        plt.tight_layout(pad=3.0)
        plt.title(config + " best grasp location for " + finger + " " + "Workspace" + " Top View") 
        ax2 = fig1.add_subplot(1,2,2,projection='3d')
        ax2.view_init(azim=30, elev=30)
        ax2.scatter(axial[0],axial[1],axial[2],marker='^',color='blue',label='Max Axial')
        ax2.scatter(radial[0],radial[1],radial[2],marker='o',color='red',label='Max Radial')
        ax2.scatter(net[0],net[1],net[2],marker='s',color='black',label='Max Net')
        ax2.scatter(self.cylinder_pts[0,:],self.cylinder_pts[1,:],self.cylinder_pts[2,:],marker='o',color='cyan',label='Cylinder')
        self.plot_hand(ax=ax2)
        plt.legend()
        plt.tight_layout(pad=3.0)
        plt.title(config + " best grasp location for " + finger + " " + "Workspace" + " Isometric View")        

    @staticmethod
    def normalize(v):
        return v / np.linalg.norm(v)

    @staticmethod
    def decompose_force(force_vector, application_point, base_point, axis_direction):
        # Normalize the axis direction
        axis_direction = Handkinematics.normalize(axis_direction)

        # Vector from base point of cylinder to the application point
        r = application_point - base_point

        # Unit vector
        r = r/Handkinematics.normalize(r)

        # Projection of r on the axis direction to find its component along the axis
        r_proj = np.dot(r, axis_direction) * axis_direction

        # Radial vector from the axis to the application point
        radial_vector = r - r_proj
        radial_direction = Handkinematics.normalize(radial_vector)

        # Tangential direction is perpendicular to both radial and axial
        tangential_direction = np.cross(axis_direction, radial_direction)

        # Decompose the force vector (If you want the force component as a vector)
        # radial_component = np.dot(force_vector, radial_direction) * radial_direction
        # axial_component = np.dot(force_vector, axis_direction) * axis_direction
        # tangential_component = np.dot(force_vector, tangential_direction) * tangential_direction

        # Force component as scalar 
        radial_component = np.dot(force_vector, radial_direction)
        axial_component = np.dot(force_vector, axis_direction)
        tangential_component = np.dot(force_vector, tangential_direction)

        return radial_component, axial_component, tangential_component

    def plot_hand(self, ax):
        """

        Plots the initial configuration of the hand prototype in 3D space.

        This function visualizes each finger of the hand in its initial joint configuration.
        Each finger is plotted as a 3D line connecting its joint positions, with different colors
        used to distinguish between fingers.

        Args:
            ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axis object where the hand configuration 
                will be plotted. Typically obtained from `matplotlib` via `fig.add_subplot(111, projection='3d')
        """

        for key in self.hand_KC.keys():
            ax.plot(self.initial_config[key][:,0],self.initial_config[key][:,1],self.initial_config[key][:,2], c=self.color[key],marker='x',label=key)
       
