# Hiwi_Robotic_Hands

This repository contains Python code for robotic hand kinematics and optimization developed as part of HiWi research at IGMR.

## Project Goal

The aim is to setup the DH parameter table for each finger of each prototype. There are namely four prototypes:-
1) MAX DOF (Both ring and little finger each have CMC joint)
2) Prototype 2 (Both ring and little finger don't have CMC joint)
3) Prototype 3 (Only little finger has CMC joint while ring finger doesn't)
4) Prototype 4 (Both ring and little finger have a common CMC joint)

First, compute the workspace of each finger using forward kinematics. Once, the forward kinematics is computed, identify the workspace intersection between the thumb and the opposing fingers. The points in the workspace intersection are then tested for the Kapandji Opposition Test and a similarity metric is computed between the thumb and the opposing finger, which compares the similarity to the ideal case for the Kapandji pose between the thumb and opposing finger. This comparison helps to establish that the additional CMC joint is better for ring and little finger from the view of the Kapandji Opposition Test. 

The next stage is to identify the best grasping locations for the cylinder for each prototype. This is done using a grasp optimization procedure which is explained in detail in Hiwi_documentation.pdf. Follow the code file pratichizzo_optimization to get the latest optimization results. The other approaches are my own formulation and implementation, and can be improved further to give more successful grasp results.

## Environment Setup

This project uses a Conda environment defined in `environment.yml`.

### Create the environment:
```bash
conda env create -f environment.yml
```

### Activate the environment:
```bash
conda activate Robotic_Hands
```

