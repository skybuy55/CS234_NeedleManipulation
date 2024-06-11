import numpy as np
#this should be processed by V-HACD before doing collision simulations.
radius = 0.1         # Base radius of the cylinder
arc_radius = 2.0        # Radius of the curvature of the arc
arc_angle = 0.5 * np.pi 
height = 2.5          # height of the needle, distributed in z, if we want flat, we should set this to 0.
resolution = 50       # Number of vertices around the circumference
segments = 120        # Number of segments along the arc

vertices = []
for j in range(segments):
    #to create the wave pattern.
    angle_shift = 0.2 * np.sin(6 * np.pi * j / segments)
    #compute the actual angle taking into account the shift as well.
    t = arc_angle * j / segments + angle_shift
    
    # Calculate center of the arc segment, the height is just constantly increasing.
    x_center = arc_radius * np.cos(t)
    y_center = arc_radius * np.sin(t)
    z_center = (height / segments) * j
    
    #adjusting the radius so each segment has a different radius.
    current_radius = radius * (1 - j / segments) * (1 + 0.25 * np.cos(8 * np.pi * j / segments))
    
    # Generate circle vertices at this segment
    for i in range(resolution):
        circle_angle = 2 * np.pi * i / resolution
        x = x_center + current_radius * np.cos(circle_angle)
        y = y_center + current_radius * np.sin(circle_angle)
        vertices.append([x, y, z_center])

vertices = np.array(vertices)

faces = []
#constructing each face with two triangles that share the edge.
for j in range(segments - 1):
    for i in range(resolution):
        #the index of the bottom left vertex
        bottom_left = j * resolution + i
        bottom_right = j * resolution + (i + 1) % resolution
        top_left = (j + 1) * resolution + i
        top_right = (j + 1) * resolution + (i + 1) % resolution
        faces.extend([[bottom_left + 1, bottom_right + 1, top_right + 1], [bottom_left + 1, top_right + 1, top_left + 1]])

with open('complex_bent_needle4.obj', 'w') as f:
    for v in vertices:
        f.write(f"v {v[0]} {v[1]} {v[2]}\n")
    for face in faces:
        f.write(f"f {face[0]} {face[1]} {face[2]}\n")