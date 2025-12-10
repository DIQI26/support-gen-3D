import os
import math
import sys
import random

from collections import defaultdict
import vedo
import numpy as np
import shapely.geometry as sh
from shapely.geometry import Polygon, LineString
from tqdm import tqdm
import warnings
from vedo import Mesh
from typing import List, Tuple, Optional, Any
from vedo import Mesh, Sphere, shapes

# Suppress RuntimeWarning related to invalid values
warnings.filterwarnings("ignore", category=RuntimeWarning)

'''
Defines a function check_intersection that takes:
stl_mesh: The 3D mesh of the stl file.
support_x: The x-coordinate of a support point.
support_y: The y-coordinate of a support point.
Goal: To check if a vertical support structure at coordinates 
(support_x, support_y) collides with the STL mesh and return the z-coordinate of the intersection point.
'''

# Define types for clarity
Point = Tuple[float, float, float]
Face = np.ndarray
Polygon = sh.Polygon
MultiPolygon = sh.MultiPolygon


def calculate_normal(p0, p1, p2):
    normal = np.cross(p1 - p0, p2 - p0)
    normal = normal / np.linalg.norm(normal)
    return normal


def calculate_torque_for_face(p0, p1, p2, centroid, weight_per_face):
    center = (p0 + p1 + p2) / 3
    force_direction = np.array([0, 0, -1])
    force = weight_per_face * force_direction
    lever_arm = center - centroid
    torque = np.cross(lever_arm, force)
    return torque, center


def calculate_all_face_torques(mesh, material_density, self_support_faces):
    weight = mesh.volume() * material_density * 9.81

    centroid = mesh.center_of_mass()
    vertices = mesh.vertices
    faces = mesh.cells
    weight_per_face = weight / len(faces)
    torques = []
    centers = []
    torque_magnitudes = []

    torsion_faces = []
    torsion_face_torques = []

    for face in faces:
        p0, p1, p2 = vertices[face]
        torque, center = calculate_torque_for_face(p0, p1, p2, centroid, weight_per_face)
        torques.append(torque)
        torque_magnitude = np.linalg.norm(torque)
        torque_magnitudes.append(torque_magnitude)
        if face in self_support_faces:
            torsion_face_torques.append(torque)
            torsion_faces.append(face)
            centers.append(center)

    average_torque = np.mean(torque_magnitudes)

    # print(f"Average torque: {average_torque:.2f}")

    return np.array(torsion_face_torques), np.array(torque_magnitudes), average_torque, torsion_faces


def add_torsion_based_support(mesh,
                              torques,
                              torsion_faces,
                              torque_magnitudes,
                              average_torque,
                              grid_spacing,
                              support_radius,
                              contact_radius,
                              extra_supports):
    faces = []

    print("\tSelecting faces that has a magnitude of torque larger than average: ")
    for i,torque in enumerate(tqdm(torques)):
        #print(torque_magnitudes[i], average_torque)
        if torque_magnitudes[i] >= average_torque * 0.8:
            faces.append(torsion_faces[i])

    if not faces:
        print("\tNo need torsion-based support!")
        return extra_supports

    projection_area = projection_merger(
        [face_projector(mesh.vertices[face]) for face in faces])

    supports_all: List[vedo.Mesh] = extra_supports
    support_coords: List[Tuple[float, float]] = []

    z_min, z_max = mesh.bounds()[4:6]

    # Get bounding box of the merged area
    min_x, min_y, max_x, max_y = projection_area.bounds

    # Equilateral triangle arrangement
    x_shift: float = (grid_spacing / 2 + support_radius)
    x_grid: float = (grid_spacing / 2 + support_radius) * math.tan(math.radians(60))
    y_grid: float = (grid_spacing / 2 + support_radius) * math.tan(math.radians(60))

    # Generate grid of points within the bounding box
    x_coords: np.ndarray = np.arange(min_x, max_x, x_grid)
    y_coords: np.ndarray = np.arange(min_y, max_y, y_grid)

    for y_idx, y_value in enumerate(y_coords):
        for x_value in x_coords:

            y_cor: float = y_value

            if y_idx % 2 == 0:
                x_cor = x_value + x_shift
            else:
                x_cor = x_value

            # Check if the location is in merged_area
            if projection_area.contains(sh.Point(x_cor, y_cor)):
                support_coords.append((x_cor, y_cor))

    triangle_list: List[vedo.shapes.Triangle] = []

    # Preprocessing: for each of the downward faces in the list,
    # create a triangle and merge them into a new mesh, use the new mesh to check the intersections
    # to reduce time complexity

    print(f"\tCreating triangles:")
    for face in tqdm(faces):
        p0, p1, p2 = mesh.vertices[face]
        triangle: vedo.shapes.Triangle = vedo.shapes.Triangle(p0, p1, p2)
        triangle_list.append(triangle)

    downward_faces_mesh: vedo.Mesh = merge_triangles_into_mesh(triangle_list)

    print(f"\tGenerate contact point spheres for cylinders:")
    for coord in tqdm(support_coords):

        # find the index of face on the mesh
        # where the original contact point belongs to by creating a vertical line at specific x and y

        intersect_faces: List[int] = downward_faces_mesh.find_cells_along_line((coord[0], coord[1], z_min - 1),
                                                                               (coord[0], coord[1], z_max + 1))

        # could intersect with multiple downward faces at the same x and y position
        for face in intersect_faces:

            # Get the vertices and create a new triangle for check intersection, which is the z value
            p0, p1, p2 = downward_faces_mesh.vertices[downward_faces_mesh.cells[face]]

            triangle: vedo.shapes.Triangle = vedo.shapes.Triangle(p0, p1, p2)
            
            normal: np.ndarray = np.cross(p1 - p0, p2 - p0)
            normal = normal / np.linalg.norm(normal)

            #face_plane_origin: np.ndarray = np.mean([p0, p1, p2], axis=0)
            intersection_top: np.ndarray = check_intersection(triangle, coord[0], coord[1])

            if intersection_top:
                # Since it is a single triangle, so there is only one intersection
                original_contact_point: float = intersection_top[0]
                # Create the original center of the sphere
                shifted_point: vedo.Point = vedo.Point(pos=(coord[0], coord[1], original_contact_point))

                support = create_single_support(mesh, triangle, shifted_point, support_radius, contact_radius)

                if support:
                    supports_all.append(support)

    return supports_all


# Check if support collides with stl and return z-coordinate of intersection
def check_intersection(stl_mesh: Mesh, support_x: float, support_y: float) -> Optional[np.ndarray]:
    line_start: Point = (support_x, support_y, stl_mesh.bounds()[4] - 1)
    line_end: Point = (support_x, support_y, stl_mesh.bounds()[5] + 1)
    intersection_points: np.ndarray = stl_mesh.intersect_with_line(line_start, line_end)

    if len(intersection_points) > 0:
        return intersection_points[:, 2]
    else:
        return None


# Takes a list of triangles as input and return a mesh object
def merge_triangles_into_mesh(triangles: List[vedo.shapes.Triangle]) -> Mesh:
    vertices: List[Point] = []
    faces: List[List[int]] = []
    idx_offset: int = 0

    for triangle in triangles:
        tri_pts: np.ndarray = triangle.vertices
        vertices.extend(tri_pts)
        faces.append([idx_offset, idx_offset + 1, idx_offset + 2])
        idx_offset += 3

    vertices_np: np.ndarray = np.array(vertices)
    faces_np: np.ndarray = np.array(faces)

    merged_mesh: Mesh = Mesh([vertices_np, faces_np])

    return merged_mesh


# Check if normal points in the negative z-direction
def face_is_downward(normal: np.ndarray) -> bool:
    return normal[2] < 0


# Check if two faces are connected to each other by sharing the same edge
def faces_are_connected(face1: Face, face2: Face) -> bool:
    # Find common point idx (3 points in face are represented by index)
    common_idx = set(face1).intersection(set(face2))

    # If two points are shared, the faces are connected by the same edge
    if len(common_idx) == 2:
        return True
    else:
        return False


# Check if point in mesh
def point_inside_mesh(point: Point, mesh: Mesh) -> bool:
    line_start: Point = (point[0], point[1], mesh.bounds()[4] - 1)
    line_end: Point = point
    intersection_points: np.ndarray = mesh.intersect_with_line(line_start, line_end)

    if len(intersection_points) == 0:
        return False
    if len(intersection_points) % 2 == 1:
        p2 = intersection_points[-1]
        if (abs(point[0] - p2[0]) <= 0.0001 and
                abs(point[1] - p2[1]) <= 0.0001 and
                abs(point[2] - p2[2]) <= 0.0001):
            return False
        else:
            return True
    else:
        return False


# Extract all faces that points downward and vertical
def faces_extractor(mesh: vedo.Mesh,
                    self_support_angle: float,
                    side_feature_size: float,
                    vertical_tolerance: float = 3.0) -> Tuple[
    List[Face], List[Face], List[Face], List[Face], List[Face]]:
    print("\nExtracting key faces...")

    # downward facing - reference to check if the face is downward
    z_axis: np.ndarray = np.array([0, 0, -1])

    downward_faces: List[Face] = []
    self_support_faces: List[Face] = []
    non_self_support_faces: List[Face] = []
    vertical_faces: List[Face] = []
    side_feature_stack: List[Face] = []

    # Extract the minimum and maximum z of the original model
    # Not height of supports, but the limitation or start/end point
    z_min: float = mesh.bounds()[4]

    # Iterate all faces and categorize them
    print(f"\tEvaluating faces:")
    for face in tqdm(mesh.cells):
        p0: np.ndarray = mesh.vertices[face[0]]
        p1: np.ndarray = mesh.vertices[face[1]]
        p2: np.ndarray = mesh.vertices[face[2]]

        normal: np.ndarray = np.cross(p1 - p0, p2 - p0)

        # Make it a unit vector
        normal = normal / np.linalg.norm(normal)

        # Skip bottom faces
        if p0[2] == z_min and p1[2] == z_min and p2[2] == z_min:
            continue

        # Extract vertical faces
        angle_to_z_axis: float = np.degrees(np.arccos(np.dot(normal, z_axis)))
        if 90 - vertical_tolerance <= angle_to_z_axis <= 90 + vertical_tolerance:
            vertical_faces.append(face)

        if face_is_downward(normal):
            # Extract all downward faces
            downward_faces.append(face)

            angle: float = np.degrees(np.arccos(np.dot(normal, z_axis)))

            # Check if the angle can be self-supported
            if angle < 90 - self_support_angle:
                non_self_support_faces.append(face)

                # Check if small enough to be a possible side feature
                vertices: np.ndarray = np.array([p0, p1, p2])
                x_size: float = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
                y_size: float = np.max(vertices[:, 1]) - np.min(vertices[:, 1])

                if x_size < side_feature_size and y_size < side_feature_size:
                    side_feature_stack.append(face)

            else:
                self_support_faces.append(face)

    # Group faces together by checking if connected to each other
    side_feature_groups: List[List[Face]] = []
    current_group: List[Face] = []
    outer_faces: List[Face] = []
    side_feature_faces: List[Face] = []

    if side_feature_stack:
        print(f"\tGrouping connected side feature faces:")
        sys.stdout.write(f"\r\tRemaining {len(side_feature_stack)} faces")
        sys.stdout.flush()

        # Search until the stack is empty
        while side_feature_stack:
            adj_faces: List[Face] = []
            if not outer_faces:
                outer_faces.append(side_feature_stack.pop(0))
            for face1 in outer_faces:
                for face2 in side_feature_stack:
                    if faces_are_connected(face1, face2):
                        if face2 not in adj_faces:
                            adj_faces.append(face2)

            current_group += outer_faces
            outer_faces = adj_faces

            if not adj_faces:
                side_feature_groups.append(current_group)
                current_group = []
                sys.stdout.write(f"\r\tRemaining: {len(side_feature_stack)} faces")
                sys.stdout.flush()

            else:
                for face in adj_faces:
                    side_feature_stack.remove(face)

        sys.stdout.write(f"\r\tRemaining: {len(side_feature_stack)} faces")
        sys.stdout.flush()

        current_group += outer_faces

        if current_group:
            side_feature_groups.append(current_group)

        print(f"\t\nMeasuring side feature face groups:")
        for face_group in tqdm(side_feature_groups):
            # Flatten the list of face vertices
            all_face_vertices: np.ndarray = np.vstack(mesh.vertices[face_group])

            # Calculate bounding box size in each direction
            x_size: float = np.max(all_face_vertices[:, 0]) - np.min(all_face_vertices[:, 0])
            y_size: float = np.max(all_face_vertices[:, 1]) - np.min(all_face_vertices[:, 1])

            if x_size < side_feature_size and y_size < side_feature_size:
                side_feature_faces.extend(face_group)

    # Ensure downward faces used for generating supports do not include side features
    for face in side_feature_faces:
        downward_faces.remove(face)

    print(f"\t{len(downward_faces)} downward-faces extracted!")
    print(f"\t{len(vertical_faces)} vertical-faces extracted!")
    print(f"\t{len(self_support_faces)} self-support-faces extracted!")
    print(f"\t{len(non_self_support_faces)} non-self-support-faces extracted!")
    print(f"\t{len(side_feature_faces)} side-feature-faces extracted!")

    return downward_faces, vertical_faces, side_feature_faces, self_support_faces, non_self_support_faces


# Project the face to the xy-plane (ignore the z-coordinate) - flattening onto x-y plane
def face_projector(face_points: np.ndarray) -> np.ndarray:
    return face_points[:, :2]


# Merge all projected area into one
def projection_merger(projected_areas: List[np.ndarray]) -> MultiPolygon:
    polygons: List[Polygon] = [sh.Polygon(area) for area in projected_areas if len(area) >= 3]
    merged_area: MultiPolygon = sh.MultiPolygon(polygons).buffer(0)
    return merged_area


# create the a inverted circular truncated cone on the contact point at bottom
def create_cone(mesh: vedo.Mesh, 
                top_radius: float, 
                bottom_radius: float, 
                x_coord: float, 
                y_coord: float, 
                contact_center_bottom_z: float,  
                line_z_top: float, 
                line_z_bottom: float):
    
    line_start = [x_coord,y_coord,line_z_top]
    line_end = [x_coord,y_coord,line_z_bottom]
    line_start = [x + y for x, y in zip(line_start, [-0.01, 0.01, 0])]
    line_end = [x + y for x, y in zip(line_end, [-0.01, 0.01, 0])]
    intersections_shift_1 = mesh.intersect_with_line(line_start, line_end)
    line_start = [x + y for x, y in zip(line_start, [0.02, -0.01, 0])]
    line_end = [x + y for x, y in zip(line_end, [0.02, -0.01, 0])]
    intersections_shift_2 = mesh.intersect_with_line(line_start, line_end)
    line_start = [x + y for x, y in zip(line_start, [-0.01, -0.01, 0])]
    line_end = [x + y for x, y in zip(line_end, [-0.01, -0.01, 0])]
    intersections_shift_3 = mesh.intersect_with_line(line_start, line_end)

    if len(intersections_shift_1) > 0 and len(intersections_shift_2) and len(intersections_shift_3) > 0:

        # find the first intersection (center contact point at bottom)
        v0 = intersections_shift_1[0]
        v1 = intersections_shift_2[0]
        v2 = intersections_shift_3[0]

        normal = np.cross(v2-v0,v1-v0)
        normal = normal / np.linalg.norm(normal)

        z_axis: np.ndarray = np.array([0,0,1])

        #print(normal, z_axis)
        angle: float = np.degrees(np.arccos(np.dot(normal,z_axis)))
        default_normal = [0,0,1]
        rotation_axis = np.cross(default_normal, normal)

        # circle at contact face
        bottom_circle = vedo.Circle(pos=(x_coord,y_coord,contact_center_bottom_z), r=bottom_radius, res=60)

        bottom_circle.rotate(angle=angle, axis=rotation_axis, point=(x_coord,y_coord,contact_center_bottom_z))

        z_up: float = top_radius / 342 * 940  # upshift for z value, about 20 degree for the angle of the cone
        cylinder_bottom_z: float = contact_center_bottom_z + z_up

        # circle at bottom of cylinder
        top_circle = vedo.Circle(pos=(x_coord,y_coord,cylinder_bottom_z), r=top_radius, res=60)

        vertices1 = top_circle.vertices
        vertices2 = bottom_circle.vertices
        vertices = np.vstack((vertices1, vertices2))
        num_points: int = top_circle.nvertices

        cone_faces: List[List[int]] = []

        for i in range(num_points):
            next_index = (i + 1) % num_points
            cone_faces.append([num_points + i, num_points + next_index, next_index, i])

        for i in range(1, num_points - 1):
            cone_faces.append([0, i, i + 1])  # bottom(contact)
            cone_faces.append(
                [num_points, num_points + i, num_points + i + 1])  # top(cylinderbottom)
            
        cone_mesh = vedo.Mesh([vertices, cone_faces])

        return cone_mesh
    
    else:
        return None

        

# create support by given xyz of contact point at top
def create_single_support(mesh: Mesh, 
                          contact_face_triangle: vedo.Triangle, 
                          contact_point_top: vedo.Point, 
                          support_radius: float, 
                          contact_point_radius: float):
    z_min = mesh.bounds()[4]
    shift_amount: float = math.sqrt(abs(support_radius ** 2 - contact_point_radius ** 2))
    v0, v1, v2 = contact_face_triangle.vertices
    normal = np.cross(v1 - v0, v2 - v0)
    normal = normal / np.linalg.norm(normal)
    contact_point_top.shift(normal * shift_amount)

    if point_inside_mesh(contact_point_top.pos(),mesh):
        return None
    contact_sphere: Sphere = vedo.Sphere(pos = contact_point_top.pos(), r=support_radius)
    
    if contact_sphere.bounds()[4] < z_min:
        return None
    
    line_start: List[float] = contact_point_top.pos()
    line_end: List[float] = [contact_point_top.pos()[0],contact_point_top.pos()[1], z_min - 1]

    # create a line from the shifted contact point to the z min of mesh, use the first intersection
    intersections_downward: np.ndarray = mesh.intersect_with_line(line_start, line_end)
    
    # initialize bottom of cylinder to z min - 0.01
    cylinder_bottom_z: float = z_min - 0.01

    # if it pass the mesh
    if len(intersections_downward) > 0:
    
        face_z_min = min(v0[2],v1[2],v2[2])

        z_shift = abs(contact_point_top.pos()[2] - face_z_min)
        
        cone = create_cone(mesh, 
                           support_radius, 
                           contact_point_radius, 
                           contact_point_top.pos()[0],
                           contact_point_top.pos()[1],
                           intersections_downward[0][2],
                           contact_point_top.pos()[2] - z_shift,
                           z_min - 1)
        
        if cone:
            #if cone.bounds()[4] > contact_point_top.pos()[2]:
                #return None
            #else:
            contact_sphere = vedo.merge(contact_sphere, cone)
            cylinder_bottom_z = cone.bounds()[5]
        else:
            return None
    
    height: float = contact_point_top.pos()[2] - cylinder_bottom_z
    if height < 0:
        return None
    mid_z: float = (height / 2) + cylinder_bottom_z

    position: Tuple[float, float, float] = (contact_point_top.pos()[0], contact_point_top.pos()[1], mid_z)
    direction: Tuple[float, float, float] = (0, 0, 1)

    cylinder: vedo.Cylinder = vedo.Cylinder(pos=position,
                                            height=height,
                                            r=support_radius,
                                            axis=direction)   

    #cylinder = cylinder.boolean(operation="-", mesh2=mesh, method=1) 
    support: vedo.Mesh = vedo.merge(contact_sphere, cylinder)

    return support


# Create support for merged projection area
def support_creator(mesh: vedo.Mesh, faces: List[List[int]], non_self_support_projection: sh.Polygon,
                    support_radius: float, grid_spacing: float, contact_radius: float) -> List[vedo.Mesh]:
    print("\nGenerating supports...")

    if not faces:
        print("Projection is empty!")
        return [],[]

    supports_all: List[vedo.Mesh] = []
    # for test output
    support_coords: List[Tuple[float, float]] = []

    z_min, z_max = mesh.bounds()[4:6]

    # Get bounding box of the merged area
    min_x, min_y, max_x, max_y = non_self_support_projection.bounds

    # Equilateral triangle arrangement
    x_shift: float = (grid_spacing / 2 + support_radius)
    x_grid: float = (grid_spacing / 2 + support_radius) * math.tan(math.radians(60))
    y_grid: float = (grid_spacing / 2 + support_radius) * math.tan(math.radians(60))

    # Generate grid of points within the bounding box
    x_coords: np.ndarray = np.arange(min_x, max_x, x_grid)
    y_coords: np.ndarray = np.arange(min_y, max_y, y_grid)

    for y_idx, y_value in enumerate(y_coords):
        for x_value in x_coords:

            y_cor: float = y_value

            if y_idx % 2 == 0:
                x_cor = x_value + x_shift
            else:
                x_cor = x_value

            # Check if the location is in merged_area
            if non_self_support_projection.contains(sh.Point(x_cor, y_cor)):
                support_coords.append((x_cor, y_cor))

    triangle_list: List[vedo.shapes.Triangle] = []

    # Preprocessing: for each of the downward faces in the list, 
    # create a triangle and merge them into a new mesh, use the new mesh to check the intersections
    # to reduce time complexity

    print(f"\tCreating triangles:")
    for face in tqdm(faces):
        p0, p1, p2 = mesh.vertices[face]
        triangle: vedo.shapes.Triangle = vedo.shapes.Triangle(p0, p1, p2)
        triangle_list.append(triangle)

    downward_faces_mesh: vedo.Mesh = merge_triangles_into_mesh(triangle_list)

    print(f"\tGenerate contact point spheres for cylinders:")
    for coord in tqdm(support_coords):

        # find the index of face on the mesh 
        # where the original contact point belongs to by creating a vertical line at specific x and y

        intersect_faces: List[int] = downward_faces_mesh.find_cells_along_line((coord[0], coord[1], z_min - 1),
                                                                               (coord[0], coord[1], z_max + 1))

        # could intersect with multiple downward faces at the same x and y position
        for face in intersect_faces:

            # Get the vertices and create a new triangle for check intersection, which is the z value
            p0, p1, p2 = downward_faces_mesh.vertices[downward_faces_mesh.cells[face]]

            triangle: vedo.shapes.Triangle = vedo.shapes.Triangle(p0, p1, p2)

            normal: np.ndarray = np.cross(p1 - p0, p2 - p0)
            normal = normal / np.linalg.norm(normal)

            #face_plane_origin: np.ndarray = np.mean([p0, p1, p2], axis=0)
            intersection_top: np.ndarray = check_intersection(triangle, coord[0], coord[1])

            if intersection_top:
                # Since it is a single triangle, so there is only one intersection
                original_contact_point: float = intersection_top[0]
                # Create the original center of the sphere
                shifted_point: vedo.Point = vedo.Point(pos=(coord[0], coord[1], original_contact_point))

                support = create_single_support(mesh, triangle, shifted_point, support_radius, contact_radius)

                if support:
                    supports_all.append(support)

    return supports_all, support_coords

# Update the calculate_surface_area_above function to calculate surface area
def calculate_surface_area_above(
        mesh: vedo.Mesh,
        slice_height: float,
        supports: List[vedo.Mesh],
        contact_point_diameter: float,
        layer_height: float,
        multiplier: float
) -> Tuple[float, int]:

    # Cut the mesh at the specified slice height to get a cross-section of the slice
    upper_mesh: vedo.Mesh = mesh.clone().cut_with_plane(origin=(0, 0, slice_height), normal=(0, 0, 1))
    lower_mesh: vedo.Mesh = mesh.clone().cut_with_plane(origin=(0, 0, (slice_height - layer_height)), normal=(0, 0, 1))

    # Create the slice by cutting the upper mesh with the lower mesh
    layer_mesh: vedo.Mesh = upper_mesh.clone().cut_with_mesh(lower_mesh, invert=True)

    # Triangulate the layer mesh to ensure it’s a solid mesh, if necessary
    layer_mesh.triangulate()

    # Calculate the surface area of the sliced portion
    slice_surface_area: float = layer_mesh.area()

    # Calculate the area of the contact point (which is a circle's area)
    contact_point_radius: float = contact_point_diameter / 2
    contact_point_area: float = math.pi * contact_point_radius ** 2

    # Set the threshold for contact points (target ratio area)
    contact_point_threshold: float = contact_point_area * multiplier

    # Find how many supports exist in the current slice height
    supports_in_slice: List[vedo.Mesh] = [
        support for support in supports
        if support.bounds()[4] <= slice_height <= support.bounds()[5]
    ]
    num_supports_in_slice: int = len(supports_in_slice)

    if num_supports_in_slice == 0:
        # If no supports are found in the current slice height
        return 0, 0  # Return 0 for both surface area per support and required additional supports

    # Calculate the surface area that each support is responsible for
    surface_area_per_support: float = slice_surface_area / num_supports_in_slice

    # Calculate the ratio with the contact point threshold
    ratio_with_x: float = surface_area_per_support / contact_point_threshold

    # Calculate the target number of supports needed to make ratio_with_x = 1
    target_num_supports: float = slice_surface_area / contact_point_threshold

    # Calculate how many more supports are needed to meet the threshold
    additional_supports_needed: int = math.ceil(target_num_supports - num_supports_in_slice)

    return ratio_with_x, additional_supports_needed


# Update add_strength_based_supports to accept strength_threshold directly
def add_strength_based_supports(
        mesh: Mesh,
        supports: List[Mesh],
        downward_faces: List[List[int]],
        area_threshold_ratio: float,
        layer_height: float,
        support_radius: float,
        contact_point_diameter: float,
        multiplier: float,
        existing_support_coords: Optional[List[Tuple[float, float]]]

) -> List[Mesh]:
    z_min, z_max = mesh.bounds()[4:6]

    extra_support_coords: List[Tuple[float, float, float]] = []  # List to hold the coordinates for extra supports
    extra_support_count: int = 0  # Counter for extra supports
    number_of_supports_required: int = 0

    current_height: float = z_min + layer_height
    contact_point_radius: float = (contact_point_diameter * math.sqrt(multiplier)) / 2

    # Define the grid size based on support radius
    grid_size = 2 * support_radius

    # Dictionary to hold supports in grid cells
    support_grid = defaultdict(list)

    # Helper function to get the grid coordinates
    def get_grid_coords(x, y):
        return (int(x // grid_size), int(y // grid_size))

    # Add existing supports to the grid
    if existing_support_coords:
        for coord in existing_support_coords:
            grid_coords = get_grid_coords(coord[0], coord[1])
            support_grid[grid_coords].append(coord)

    # Generate extra support coordinates based on surface area per support
    while current_height < z_max:
        area_ratio, number_of_supports_required = calculate_surface_area_above(
            mesh, current_height, supports, contact_point_diameter, layer_height, multiplier
        )

        if area_ratio > area_threshold_ratio:
            added_supports: int = 0
            while added_supports < number_of_supports_required:
                face: List[int] = random.choice(downward_faces)
                p0, p1, p2 = mesh.vertices[face]

                avg_x: float = (p0[0] + p1[0] + p2[0]) / 3
                avg_y: float = (p0[1] + p1[1] + p2[1]) / 3

                angle: float = random.uniform(0, 2 * math.pi)
                offset_x: float = (2 * contact_point_radius + 0.01) * math.cos(angle)
                offset_y: float = (2 * contact_point_radius + 0.01) * math.sin(angle)

                new_x: float = avg_x + offset_x
                new_y: float = avg_y + offset_y
                new_point: Tuple[float, float, float] = (new_x, new_y, current_height)

                # Get grid coordinates for the new point
                grid_coords = get_grid_coords(new_x, new_y)

                # Check for overlap with existing supports in the same and neighboring cells
                overlap_found = False
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        neighbor_cell = (grid_coords[0] + i, grid_coords[1] + j)
                        for existing_point in support_grid[neighbor_cell]:
                            dist = math.sqrt((new_x - existing_point[0]) ** 2 + (new_y - existing_point[1]) ** 2)
                            if dist < 2 * support_radius:
                                overlap_found = True
                                break
                        if overlap_found:
                            break
                    if overlap_found:
                        break

                # Skip this support if overlap is found
                if overlap_found:
                    continue

                # Add the new point if it passes the proximity check
                extra_support_coords.append(new_point)
                support_grid[grid_coords].append(new_point[:2])  # Add the point's 2D position to the grid

                extra_support_count += 1
                added_supports += 1

        current_height += layer_height

    if extra_support_count == 0:
        print("\tNo need strength-based support!")
        return []

    #print(f"Total number of extra strength-based supports added: {extra_support_count}")

    # Preprocessing downward faces to merge into a new mesh, same logic as support_creator
    print(f"\tCreating triangles:")
    triangle_list: List[shapes.Triangle] = []
    for face in tqdm(downward_faces):
        p0, p1, p2 = mesh.vertices[face]
        triangle: shapes.Triangle = vedo.shapes.Triangle(p0, p1, p2)
        triangle_list.append(triangle)

    downward_faces_mesh: Mesh = merge_triangles_into_mesh(triangle_list)

    supports_all: List[Mesh] = []
    print(f"\tGenerate contact point spheres and cylinders:")

    for coord in tqdm(extra_support_coords):

        intersect_faces: List[int] = downward_faces_mesh.find_cells_along_line(
            (coord[0], coord[1], z_min - 1),
            (coord[0], coord[1], z_max + 1)
        )

        for face in intersect_faces:
            p0, p1, p2 = downward_faces_mesh.vertices[downward_faces_mesh.cells[face]]
            triangle = vedo.shapes.Triangle(p0, p1, p2)
            normal = np.cross(p1 - p0, p2 - p0)
            normal = normal / np.linalg.norm(normal)

            intersection_top: Optional[np.ndarray] = check_intersection(triangle, coord[0], coord[1])

            if intersection_top:
                original_contact_point: float = intersection_top[0]
                shifted_point: Point = vedo.Point(pos=(coord[0], coord[1], original_contact_point))
            
                support = create_single_support(mesh, triangle, shifted_point, support_radius, contact_point_diameter/2)

                if support:
                    supports_all.append(support)

    return supports_all


# Handle and return CUI input data
def input_handler(input_file: str) -> Tuple[Tuple[str, str], float, float, float, float, float, float, float, float]:
    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    file_path: str = os.path.join(script_dir, "src", input_file)
    file_type: str = "STL"

    # Preload the size of file
    mesh_bound: Any = vedo.Mesh(file_path).bounds()
    size_x: float = mesh_bound[1] - mesh_bound[0]
    size_y: float = mesh_bound[3] - mesh_bound[2]
    size_z: float = mesh_bound[5] - mesh_bound[4]
    print(f"Input file: {input_file}")
    print(f"File type: {file_type}")
    print(f"Dimension: {size_x:.2f} * {size_y:.2f} * {size_z:.2f}")

    # Variable declarations with default values
    maximum_column_diameter: float
    spacing_between_supports: float
    self_support_angle: float
    spacing_from_model: float
    contact_point_diameter: float
    side_feature_size: float
    weight_multiplier: float
    material_density: float

    # 8 while loops for 8 parameters
    while True:
        try:
            p1_input: str = input(f"\n1. Please input the maximum column diameter "
                                  f"(default {math.sqrt((size_x * size_y) / 500) / 2:.2f}): ").strip()
            if p1_input == "":
                # An equation to estimate the size of one support by approximating 500 positions of supports
                maximum_column_diameter = math.sqrt((size_x * size_y) / 500) / 2
                print(f"No input given! Using default.")
                break
            maximum_column_diameter = float(p1_input)
            if maximum_column_diameter <= 0:
                print(f"Error input '{p1_input}': Must be a non-zero positive number!")
            else:
                break
        except ValueError:
            print(f"Error input '{p1_input}': Please enter a valid number!")

    while True:
        try:
            p2_input: str = input(f"\n2. Please input the spacing between supports "
                                  f"(default {maximum_column_diameter:.2f}): ").strip()
            if p2_input == "":
                spacing_between_supports = maximum_column_diameter
                print(f"No input given! Using default.")
                break
            spacing_between_supports = float(p2_input)
            if spacing_between_supports <= 0:
                print(f"Error input '{p2_input}': Must be a non-zero positive number!")
            else:
                break
        except ValueError:
            print(f"Error input '{p2_input}': Please enter a valid number!")

    while True:
        try:
            p3_input: str = input("\n3. Please input the self-support angle "
                                  "(default 20): ").strip()
            if p3_input == "":
                self_support_angle = 20
                print(f"No input given! Using default.")
                break
            self_support_angle = float(p3_input)
            if self_support_angle < 0 or self_support_angle > 90:
                print(f"Error input '{p3_input}': Must be a number between 0 and 90!")
            else:
                break
        except ValueError:
            print(f"Error input '{p3_input}': Please enter a valid number!")

    while True:
        try:
            p4_input: str = input(f"\n4. Please input the spacing from model "
                                  f"(default {maximum_column_diameter / 4:.2f}): ").strip()
            if p4_input == "":
                spacing_from_model = maximum_column_diameter / 4
                print(f"No input given! Using default.")
                break
            spacing_from_model = float(p4_input)
            if spacing_from_model < 0:
                print(f"Error input '{p4_input}': Must be a non-zero positive number!")
            else:
                break
        except ValueError:
            print(f"Error input '{p4_input}': Please enter a valid number!")

    while True:
        try:
            p5_input: str = input(f"\n5. Please input the contact point diameter "
                                  f"(default {maximum_column_diameter / 2:.2f}): ").strip()
            if p5_input == "":
                contact_point_diameter = maximum_column_diameter / 2
                print(f"No input given! Using default.")
                break
            contact_point_diameter = float(p5_input)
            if contact_point_diameter <= 0:
                print(f"Error input '{p5_input}': Must be a non-zero positive number!")
            elif contact_point_diameter > maximum_column_diameter:
                print(f"Error input '{p5_input}': Must be smaller than the maximum column diameter!")
            else:
                break
        except ValueError:
            print(f"Error input '{p5_input}': Please enter a valid number!")

    while True:
        try:
            p6_input: str = input("\n6. Please input the maximum side feature size (default 0): ").strip()
            if p6_input == "":
                side_feature_size = 0
                print(f"No input given! Using default.")
                break
            side_feature_size = float(p6_input)
            if side_feature_size < 0:
                print(f"Error input '{p6_input}': Must be a positive number!")
            else:
                break
        except ValueError:
            print(f"Error input '{p6_input}': Please enter a valid number!")

    while True:
        try:
            p7_input: str = input("\n7. Please input the weight multiplier (default 10): ").strip()
            if p7_input == "":
                weight_multiplier = 10
                print(f"No input given! Using default.")
                break
            weight_multiplier = float(p7_input)
            if weight_multiplier <= 0:
                print(f"Error input '{p7_input}': Must be a non-zero positive number!")
            else:
                break
        except ValueError:
            print(f"Error input '{p7_input}': Please enter a valid number!")


    while True:
        try:
            p8_input: str = input("\n8. Please input the material density in kg/m³ (default 1000): ").strip()
            if p8_input == "":
                material_density = 1000
                print(f"No input given! Using default.")
                break
            material_density = float(p8_input)
            if material_density <= 0:
                print(f"Error input '{p8_input}': Must be a non-zero positive number!")
            else:
                break
        except ValueError:
            print(f"Error input '{p8_input}': Please enter a valid number!")

    return (file_path,
            file_type), \
           spacing_between_supports, \
           maximum_column_diameter, \
           self_support_angle, \
           spacing_from_model, \
           contact_point_diameter, \
           side_feature_size, \
           weight_multiplier, \
           material_density


def result_saver(file_path: str, combined_mesh) -> None:
    # Get the directory of the script
    script_dir: str = os.path.dirname(os.path.abspath(__file__))

    # Extract filename for output name
    base_name: str = os.path.basename(file_path)
    file_name: Optional[str] = os.path.splitext(base_name)[0]

    # If file_name is still None, set a default name
    if not file_name:
        file_name = "output"

    # Define the output folder path relative to the script's directory
    folder_path: str = os.path.join(script_dir, "output")
    output_file_name: str = f"{file_name}_output.stl"
    output_path: str = os.path.join(folder_path, output_file_name)

    saving: bool = True
    while saving:
        save: str = input("\nDo you want to save this result? [Y/N]: ").strip().upper()

        if save == "Y":
            # Create folder if it does not exist
            os.makedirs(folder_path, exist_ok=True)

            # Save the result
            combined_mesh.write(output_path)
            print(f"Result saved at: {output_path}")
            saving = False

        elif save == "N":
            print("Result aborted!")
            saving = False

        else:
            print("Please enter Y or N!")


# Check if the z-axis of one face and one support overlaps
# spacing from model
def face_overlaps_support_on_z(face_vertices: np.ndarray, support: Any, support_radius: float) -> bool:
    # Extract the highest and lowest z coordinates for face and support
    face_z_min: float = np.min(face_vertices[:, 2]) + support_radius
    face_z_max: float = np.max(face_vertices[:, 2]) - support_radius
    support_z_min: float = support.bounds()[4]
    support_z_max: float = support.bounds()[5]

    # Check for overlap in the z-direction
    if (face_z_min < support_z_min < face_z_max or
            face_z_min < support_z_max < face_z_max or
            support_z_min < face_z_min < support_z_max or
            support_z_min < face_z_max < support_z_max or
            (support_z_min == face_z_min and support_z_max == face_z_max)):
        return True
    return False


# Filter the support generated based on side_feature_size & spacing_from_model
def support_filter(
        mesh: Any,
        vertical_faces: List[np.ndarray],
        supports: List[Any],
        spacing_from_model: float,
        support_radius: float
) -> List[Any]:
    print("\nFiltering supports...")

    # List to store supports that pass the filtering
    filtered_supports: List[Any] = []

    # Loop through each support structure
    for support in tqdm(supports):
        # Skip if the support is generated inside the mesh
        if point_inside_mesh(support.center_of_mass(), mesh):
            continue

        too_close: bool = False

        # Create a bounding box polygon for the support in the XY plane
        support_bounds: np.ndarray = support.bounds()

        # Iterate through each vertical face
        for face in vertical_faces:
            # Check if the support and face overlap on the z-axis
            if face_overlaps_support_on_z(mesh.vertices[face], support, support_radius):
                # Project the face vertices to a polygon in the XY plane
                face_polygon: Polygon = Polygon(mesh.vertices[face][:, :2])

                # Extract the vertex coordinates of the selected face
                v1, v2, v3 = mesh.vertices[face]

                # Project the support to a line in the XY plane
                support_line: LineString = LineString([
                    (support_bounds[0], support_bounds[2], min(v1[2], v2[2], v3[2])),  # (min_x, min_y, min_z)
                    (support_bounds[1], support_bounds[3], max(v1[2], v2[2], v3[2]))  # (max_x, max_y, max_z)
                ])

                # Calculate the horizontal distance between the face and the support line
                horizontal_distance: float = face_polygon.distance(support_line)

                # If the support is too close to the face, mark it as too close and stop checking this support
                if horizontal_distance < spacing_from_model:
                    too_close = True
                    break

        # Add the support to the filtered list if it passed the distance check
        if not too_close:
            filtered_supports.append(support)

    print(f"\tNumber of supports after filtering: {len(filtered_supports)}")

    return filtered_supports


# Show visualize window for the source mesh and result mesh
def result_visualizer(
        mesh: Any,
        combined_mesh: Any,
        vertical_faces: List[int],
        non_self_support_faces: List[int],
        self_support_faces: List[int],
        side_feature_faces: List[int],
        supports,
        extra_supports
) -> None:
    print("\nVisualization window is ready!")

    # Initialize two containers for plotting result and source meshes
    plt_result: List[Any] = [combined_mesh]
    plt_source: List[Any] = [mesh]

    # Create and color meshes for different types of faces if available
    if len(side_feature_faces) > 0:
        side_feature_mesh: Any = vedo.Mesh([mesh.vertices, side_feature_faces]).color("green").opacity(1)
        plt_result.append(side_feature_mesh)
        plt_source.append(side_feature_mesh)

    if len(vertical_faces) > 0:
        vertical_mesh: Any = vedo.Mesh([mesh.vertices, vertical_faces]).color("red").opacity(1)
        plt_result.append(vertical_mesh)
        plt_source.append(vertical_mesh)

    # Purple components
    if len(non_self_support_faces) > 0:
        non_self_support_mesh: Any = vedo.Mesh([mesh.vertices, non_self_support_faces]).color("purple").opacity(1)
        plt_result.append(non_self_support_mesh)
        plt_source.append(non_self_support_mesh)

    if len(supports) > 0:
        regular_mesh = vedo.merge(*supports).color("purple").opacity(1)
        plt_result.append(regular_mesh)

    if len(self_support_faces) > 0:
        self_support_mesh: Any = vedo.Mesh([mesh.vertices, self_support_faces]).color("pink").opacity(1)
        plt_result.append(self_support_mesh)
        plt_source.append(self_support_mesh)

    # Pink components
    if len(extra_supports) > 0:
        extra_mesh = vedo.merge(*extra_supports).color("pink").opacity(1)
        plt_result.append(extra_mesh)

    # Create plotter and display the meshes
    plt: vedo.Plotter = vedo.Plotter(shape=(2, 1), axes=7)
    plt.at(0).show(plt_result, "Result")
    plt.at(1).show(plt_source, "Source")
    plt.interactive().close()


def stl_main(input_file: str) -> None:
    
    supports = []
    extra_supports = []
    # Parse input parameters
    user_input: List[Any] = input_handler(input_file)
    file_path: str = user_input[0][0]
    file_type: str = user_input[0][1]
    spacing_between_supports: float = user_input[1]
    maximum_column_diameter: float = user_input[2]
    self_support_angle: float = user_input[3]
    spacing_from_model: float = user_input[4]
    contact_point_diameter: float = user_input[5]
    side_feature_size: float = user_input[6]
    weight_multiplier: float = user_input[7]
    material_density: float = user_input[8] # ensity of PLA in kg/m³

    strength_threshold: float = contact_point_diameter * weight_multiplier
    layer_height: float = 2.0
    support_radius: float = maximum_column_diameter / 2

    # Load file as vedo mesh object
    mesh: Any = vedo.Mesh(file_path)

    # Extract faces from mesh and categorize them
    faces_result: List[List[int]] = faces_extractor(mesh, self_support_angle, side_feature_size)
    downward_faces: List[int] = faces_result[0]  # Any downward faces
    vertical_faces: List[int] = faces_result[1]  # Vertical faces
    side_feature_faces: List[int] = faces_result[2]  # Side feature faces
    self_support_faces: List[int] = faces_result[3]  # Self-supporting faces
    non_self_support_faces: List[int] = faces_result[4]  # Non-self-supporting faces

    # Preview key faces
    print("\nPreview faces...")
    print("\tPurple: Faces that CANNOT support themselves. Need regular supports.")
    print("\tPink: Faces that CAN support themselves. May Need extra weight/torsion supports.")
    print("\tRed: Faces that are VERTICAL. Supports too close would be filtered.")
    print("\tGreen: Faces that belong to a SIDE FEATURE. Avoid adding any support on them.")
    print("\tYellow: Faces that are not labelled with any type above.")
    result_visualizer(mesh,
                      mesh.copy(),
                      vertical_faces,
                      non_self_support_faces,
                      self_support_faces,
                      side_feature_faces,
                      [],
                      [])

    # Error handling for 0 face need support
    if len(downward_faces) == 0:
        print("\nNo support needed!")
        combined_mesh: Any = mesh
    else:
        # Find projection areas & merge all areas into one
        non_self_support_projection: Any = projection_merger(
            [face_projector(mesh.vertices[face]) for face in non_self_support_faces])

        supports: List[Any]
        support_coords: List[Any]

        # Generate support structures based on merged area
        supports, support_coords = support_creator(
            mesh,
            non_self_support_faces,
            non_self_support_projection,
            support_radius,
            spacing_between_supports,
            contact_point_diameter / 2
        )

        # Filter support structures based on side_feature_size and spacing_from_model
        if supports:
            supports = support_filter(mesh, vertical_faces, supports, spacing_from_model, maximum_column_diameter / 2)

        # Add strength-based supports
        print("\nAdding extra strength-based supports...")
        extra_supports: List[Any] = add_strength_based_supports(
            mesh,
            supports,
            downward_faces,
            strength_threshold,
            layer_height,
            support_radius,
            contact_point_diameter,
            weight_multiplier,
            support_coords
        )

        # Add strength-based supports
        print("\nAdding extra torsion-based supports...")
        torques, torque_magnitudes, avg_torque, torsion_faces = calculate_all_face_torques(mesh,
                                                                                           material_density,
                                                                                           self_support_faces)

        # Add torsion-based supports

        extra_supports: List[Any] = add_torsion_based_support(mesh,
                                                              torques,
                                                              torsion_faces,
                                                              torque_magnitudes,
                                                              avg_torque,
                                                              spacing_between_supports,
                                                              support_radius,
                                                              contact_point_diameter / 2,
                                                              extra_supports)

        # Merge supports with the original mesh
        print("\nMerging mesh...")
        combined_mesh = vedo.merge(mesh, supports, *extra_supports)

    # Show the result
    result_visualizer(mesh,
                      combined_mesh,
                      vertical_faces,
                      non_self_support_faces,
                      self_support_faces,
                      side_feature_faces,
                      supports,
                      extra_supports)

    # Decide if save result
    result_saver(file_path, combined_mesh)


if __name__ == "__main__":
    stl_main("inclined_cuboid_53.stl")

