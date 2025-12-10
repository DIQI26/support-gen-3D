import os
import math
import vedo
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from tqdm import tqdm
from typing import List, Optional


class SLCReader:
    LAST_CONTOUR: int = 0xffffffff
    MAXIMUM_HEADER_SIZE: int = 2048
    MAXIMUM_VERTEX_SIZE: int = 0x1fffffff

    def __init__(self, file_path: Path, scale: float = 1.0, offset: np.ndarray = np.array([0.0, 0.0, 0.0])):
        '''
        Initializes the SLCReader with a file path, scale, and offset.
        
        :param file_path: The path to the SLC file.
        :param scale: The scale factor for resizing contours.
        :param offset: A 3D offset to apply to the contours.
        '''
        self.file_path: Path = file_path
        self.stream = open(file_path, 'rb')
        self.header: List[str] = []
        self.next_contour_count: int = 0
        self.next_z: float = np.nan
        self.scale: float = scale
        self.offset: np.ndarray = offset

    def device_readable(self) -> bool:
        '''
        Checks if the stream is readable.

        :return: True if the stream is readable, False otherwise.
        '''
        return self.stream.readable()

    def read_header(self) -> bool:
        '''
        Reads the header of the SLC file to retrieve metadata.

        :return: True if the header was successfully read, False otherwise.
        '''
        if not self.header:
            if self.stream.read(1) == b'':  
                print("The SLC file is empty or corrupt.")  
            raw_header: bytearray = bytearray()
            ch: bytes = b' '

            while True:
                line: bytes = self.stream.readline()
                if not line:
                    raw_header.clear()
                    break
                raw_header += ch + line.strip() + b' '
                ch = self.stream.read(1)
                if ch == b'\x1a':
                    break

            self.header = raw_header.decode('utf-8').split()
            if not self.header or ch != b'\x1a':
                return False

            id_type: int = self.header.index('-TYPE') + 1 if '-TYPE' in self.header else -1
            if id_type > 0 and id_type < len(self.header) and self.header[id_type].strip() == 'WEB':
                return False

            id_unit: int = self.header.index('-UNIT') + 1 if '-UNIT' in self.header else -1
            if id_unit > 0 and id_unit < len(self.header) and self.header[id_unit].strip() == 'INCH':
                self.scale = 25.4

            self.stream.seek(256, 1)
            byte: int = int.from_bytes(self.stream.read(1), 'little')
            self.stream.seek(4 * 4 * byte, 1)
            self.next_z = np.frombuffer(self.stream.read(4), dtype=np.float32)[0]
            self.next_contour_count = int.from_bytes(self.stream.read(4), 'little')

        return True

    def next_z_value(self) -> float:
        '''
        Returns the next Z value if it can be read from the stream.
        
        :return: The next Z value or NaN if it cannot be read.
        '''
        if self.device_readable() and self.next_contour_count != SLCReader.LAST_CONTOUR and self.read_header():
            return self.next_z + self.offset[2]
        else:
            return np.nan

    def read_slice(self, join_gaps: bool = False) -> Optional[dict]:
        '''
        Reads a single slice (layer) of the SLC file.
        
        :param join_gaps: Whether to join gaps in the contour.
        :return: A dictionary containing the slice data (contours and thickness) or None if there are no more slices.
        '''
        if not self.device_readable() or self.next_contour_count == SLCReader.LAST_CONTOUR or not self.read_header():
            return None
        
        if self.next_contour_count == 0:
            return None

        slice_data: dict = {
            'z': self.next_z,
            'contours': [],
            'thickness': np.nan
        }

        for _ in range(self.next_contour_count):
            vertex_count: int = int.from_bytes(self.stream.read(4), 'little')
            if vertex_count == 0:
                continue
            gap_count: int = int.from_bytes(self.stream.read(4), 'little')

            if vertex_count:
                vertices: np.ndarray = np.frombuffer(
                    self.stream.read(min(vertex_count, SLCReader.MAXIMUM_VERTEX_SIZE) * 8),
                    dtype=np.float32
                ).reshape(-1, 2)

                if join_gaps and gap_count:
                    vertices = self._join_gaps(vertices)

                if len(vertices) > 1 and np.array_equal(vertices[0], vertices[-1]):
                    vertices = vertices[:-1]

                slice_data['contours'].append(self._scale(vertices))

        self.next_z = np.frombuffer(self.stream.read(4), dtype=np.float32)[0]
        self.next_contour_count = int.from_bytes(self.stream.read(4), 'little')
        slice_data['thickness'] = self.next_z - slice_data['z']

        if not np.isclose(self.scale, 1.0):
            slice_data['z'] *= self.scale
            slice_data['thickness'] *= self.scale

            adjusted_contours: List[np.ndarray] = []
            for contour in slice_data['contours']:
                adjusted_contour = contour + self.offset[:2]
                adjusted_contours.append(adjusted_contour)

            slice_data['contours'] = adjusted_contours
        return slice_data

    def _scale(self, vertices: np.ndarray) -> np.ndarray:
        '''
        Scales the vertices according to the scale factor.
        
        :param vertices: The input vertices to scale.
        :return: The scaled vertices.
        '''
        if np.isclose(self.scale, 1.0) or not vertices.size:
            return vertices
        return vertices * self.scale

    def _join_gaps(self, vertices: np.ndarray) -> np.ndarray:
        '''
        Joins gaps in the contour by removing consecutive repeated vertices.
        
        :param vertices: The vertices of the contour.
        :return: A new array of vertices with gaps joined.
        '''
        gap_free: List[np.ndarray] = []
        from_idx: int = 0

        for j in range(1, len(vertices)):
            if np.array_equal(vertices[j - 1], vertices[j]):
                gap_free.extend(vertices[from_idx:j])
                from_idx = j
        gap_free.extend(vertices[from_idx:])
        return np.array(gap_free)

    def read_slices(self, join_gaps: bool = False) -> List[dict]:
        '''
        Reads all slices (layers) from the SLC file.
        
        :param join_gaps: Whether to join gaps in each slice.
        :return: A list of all slices as dictionaries.
        '''
        all_slices: List[dict] = []
        while True:
            slice_data = self.read_slice(join_gaps)
            if slice_data is None:
                break
            all_slices.append(slice_data)
        return all_slices

    def close(self) -> None:
        '''
        Closes the stream associated with the SLC file.
        '''
        self.stream.close()


class SupportParameter:
    def __init__(self, spacing_between_supports: float, maximum_column_diameter: float, self_support_angle: float,
                 spacing_from_model: float, contact_point_diameter: float):
        '''
        Initializes the parameters for generating supports.
        '''
        self.spacing_between_supports = spacing_between_supports
        self.maximum_column_diameter = maximum_column_diameter
        self.self_support_angle = self_support_angle
        self.spacing_from_model = spacing_from_model
        self.contact_point_diameter = contact_point_diameter
        self.support_radius = maximum_column_diameter / 2

    def __str__(self):
        '''
        Returns a string representation of the support parameters.
        '''
        return (f"SupportParameter(spacing_between_supports={self.spacing_between_supports}, "
                f"maximum_column_diameter={self.maximum_column_diameter}, self_support_angle={self.self_support_angle}, "
                f"spacing_from_model={self.spacing_from_model}, contact_point_diameter={self.contact_point_diameter})")


class SupportParameterBuilder:
    def __init__(self):
        # Default values
        self.spacing_between_supports = 0.5
        self.maximum_column_diameter = 0.2
        self.self_support_angle = 20
        self.spacing_from_model = 0.5
        self.contact_point_diameter = 0.2

    def set_spacing_between_supports(self, spacing: float) -> 'SupportParameterBuilder':
        '''
        Sets the spacing between supports.
        '''
        self.spacing_between_supports = spacing
        return self

    def set_maximum_column_diameter(self, diameter: float) -> 'SupportParameterBuilder':
        '''
        Sets the maximum column diameter.
        '''
        self.maximum_column_diameter = diameter
        return self

    def set_self_support_angle(self, angle: float) -> 'SupportParameterBuilder':
        '''
        Sets the self-support angle.
        '''
        self.self_support_angle = angle
        return self

    def set_spacing_from_model(self, spacing: float) -> 'SupportParameterBuilder':
        '''
        Sets the spacing from the model.
        '''
        self.spacing_from_model = spacing
        return self

    def set_contact_point_diameter(self, diameter: float) -> 'SupportParameterBuilder':
        '''
        Sets the contact point diameter.
        '''
        self.contact_point_diameter = diameter
        return self

    def build(self) -> SupportParameter:
        '''
        Builds the SupportParameter object.
        '''
        return SupportParameter(
            spacing_between_supports=self.spacing_between_supports,
            maximum_column_diameter=self.maximum_column_diameter,
            self_support_angle=self.self_support_angle,
            spacing_from_model=self.spacing_from_model,
            contact_point_diameter=self.contact_point_diameter
        )


class SupportGenerator:
    def __init__(self, slices: List[dict], merged_area: Polygon, settings: SupportParameter):
        '''
        Initializes the SupportGenerator for creating support structures.
        
        :param slices: The slices of the model as a list of dictionaries.
        :param merged_area: The merged polygon area of the model.
        :param settings: The settings used for support generation.
        '''
        self.slices: List[dict] = slices
        self.merged_area: Polygon = merged_area
        self.settings: SupportParameter = settings
        self.existing_support_points: List[List[float]] = []
        self.any_support_needed: bool = False

    def create_support_for_angle(self) -> List[vedo.Cylinder]:
        '''
        Creates supports based on the self-support angle for the model.
        
        :return: A list of `vedo.Cylinder` objects representing the supports.
        '''
        supports_for_angle: List[vedo.Cylinder] = []

        if not self.slices:
            return supports_for_angle

        max_z: float = self.slices[-1]['z']
        min_z: float = self.slices[0]['z']
        offset_area: Polygon = self.merged_area.buffer(-self.settings.spacing_from_model)

        if offset_area.is_empty:
            return supports_for_angle

        for slice_data in tqdm(self.slices, desc="Calculating angle-based supports", unit="slice"):
            for contour in slice_data['contours']:
                angle: float = calculate_angle_with_vertical(contour)

                if angle > self.settings.self_support_angle:
                    self.any_support_needed = True
                    centroid: np.ndarray = np.mean(contour, axis=0)

                    intersections: List[float] = check_intersection(self.slices, centroid[0], centroid[1])

                    if intersections:
                        intersections = np.insert(intersections, 0, min_z)
                        supports_for_angle.extend(self._create_cylindrical_supports(centroid, intersections))

        return supports_for_angle

    def create_support_for_parameters(self) -> List[vedo.Cylinder]:
        '''
        Creates supports based on a grid of parameters, like spacing between supports.
        
        :return: A list of `vedo.Cylinder` objects representing the supports.
        '''
        supports_for_params: List[vedo.Cylinder] = []
        if not self.any_support_needed:
            return supports_for_params

        min_x, min_y, max_x, max_y = self.merged_area.buffer(-self.settings.spacing_from_model).bounds
        #hexagonal pattern
        x_grid: float = self.settings.spacing_between_supports / 2
        y_grid: float = (self.settings.spacing_between_supports / 2) * math.tan(math.radians(60))

        x_coords: np.ndarray = np.arange(min_x, max_x, x_grid)
        y_coords: np.ndarray = np.arange(min_y, max_y, y_grid)

        total_iterations: int = len(x_coords) * len(y_coords)

        with tqdm(total=total_iterations, desc="Calculating parameter-based supports", unit="grid point") as pbar:
            for index, x in enumerate(x_coords):
                for y_cor in y_coords:
                    #for staggered grid
                    x_cor: float = x if index % 2 else x + x_grid / 2
                    point = Point(x_cor, y_cor)

                    if self.merged_area.contains(point):
                        if not supports_overlap(self.existing_support_points, [x_cor, y_cor], self.settings.support_radius * 2):
                            intersections: List[float] = check_intersection(self.slices, x_cor, y_cor)
                            if intersections:
                                intersections = np.insert(intersections, 0, self.slices[0]['z'])
                                supports_for_params.extend(self._create_cylindrical_supports([x_cor, y_cor], intersections))
                    pbar.update(1)

        return supports_for_params

    def _create_cylindrical_supports(self, centroid: np.ndarray, intersections: List[float]) -> List[vedo.Cylinder]:
        '''
        Creates cylindrical supports at the given centroid based on intersections.
        
        :param centroid: The centroid (x, y) of the support.
        :param intersections: A list of Z values where intersections occur.
        :return: A list of `vedo.Cylinder` objects representing the supports.
        '''
        supports: List[vedo.Cylinder] = []
        #the top intersection is processed only when idx is odd
        for idx, z_cor in enumerate(intersections):
            if idx % 2 == 1:
                passed_z_cor: float = intersections[idx - 1]
                height: float = z_cor - passed_z_cor - 2 * (self.settings.contact_point_diameter / 2)
                mid_z: float = (height / 2) + passed_z_cor
                if height > 0 and not supports_overlap(self.existing_support_points, centroid, self.settings.support_radius * 2):
                    support = self.create_contact_point(centroid, mid_z, height)
                    supports.append(support)
        return supports

    def create_contact_point(self, centroid: np.ndarray, mid_z: float, height: float) -> vedo.Cylinder:
        '''
        Creates a cylindrical support structure, including top and bottom contact points.
        
        :param centroid: The centroid of the support.
        :param mid_z: The midpoint Z value of the support.
        :param height: The height of the support cylinder.
        :return: A `vedo.Cylinder` object representing the support structure.
        '''
        cylinder = vedo.Cylinder(
            pos=(centroid[0], centroid[1], mid_z),
            height=height,
            r=self.settings.support_radius,
            axis=(0, 0, 1)
        )
        
        top_sphere = vedo.Sphere(
            pos=(centroid[0], centroid[1], mid_z + height / 2),
            r=self.settings.contact_point_diameter / 2
        )
        
        ground_z = self.slices[0]['z']
        if np.isclose(mid_z - height / 2, ground_z, atol=1e-3):
            bottom_contact = vedo.Cylinder(
                pos=(centroid[0], centroid[1], mid_z - height / 2),
                height=self.settings.contact_point_diameter, 
                r=self.settings.maximum_column_diameter / 2,
                axis=(0, 0, 1)
            )
        else:
            bottom_contact = vedo.Sphere(
                pos=(centroid[0], centroid[1], mid_z - height / 2),
                r=self.settings.contact_point_diameter / 2
            )
        support_structure = vedo.merge(cylinder, top_sphere, bottom_contact)
        
        self.existing_support_points.append([centroid[0], centroid[1]])
        return support_structure

def calculate_angle_with_vertical(contour: np.ndarray) -> float:
    '''
    Calculates the average angle between a contour and the vertical axis using arctan2.
    
    :param contour: The contour as an array of (x, y) vertices.
    :return: The average angle between the contour and the vertical axis.
    '''
    angles: List[float] = []

    for i in range(len(contour) - 1):
        p1, p2 = contour[i], contour[i + 1]
        dx = p2[0] - p1[0]  #difference in x
        dy = p2[1] - p1[1]  #difference in y

        if dx == 0 and dy == 0:
            continue  
        angle = np.degrees(np.arctan2(dx, dy)) 
        angle = abs(angle)
        if angle > 90:
            angle = 180 - angle

        angles.append(angle)

    average_angle = np.mean(angles) if angles else 0
    return average_angle


def supports_overlap(existing_support_points: List[List[float]], new_point: List[float], min_distance: float) -> bool:
    '''
    Checks if a new support point overlaps with any existing support points.
    
    :param existing_support_points: The list of existing support points.
    :param new_point: The new support point to check.
    :param min_distance: The minimum allowable distance between supports.
    :return: True if the new point overlaps with existing points, False otherwise.
    '''
    #euclidean distance is calculated
    return any(np.linalg.norm(np.array(existing_point) - np.array(new_point)) < min_distance for existing_point in existing_support_points)


def merge_polygons(polygons: List[Polygon]) -> Optional[Polygon]:
    '''
    Merges a list of polygons into a single polygon.
    
    :param polygons: The list of polygons to merge.
    :return: A merged polygon or None if no valid polygons exist.
    '''
    valid_polygons: List[Polygon] = [poly for poly in polygons if poly.is_valid and not poly.is_empty]
    return unary_union(valid_polygons) if valid_polygons else None


def check_intersection(slices: List[dict], support_x: float, support_y: float) -> List[float]:
    '''
    Checks for intersections between a support point and the contours in each slice.
    
    :param slices: The slices containing contours.
    :param support_x: The X-coordinate of the support point.
    :param support_y: The Y-coordinate of the support point.
    :return: A list of Z values where the support intersects with contours.
    '''
    support_point: Point = Point(support_x, support_y)
    intersection_z_values: List[float] = []

    for s in slices:
        z_value: float = s['z']
        contours: List[np.ndarray] = s['contours']

        found = False
        for contour in contours:
            polygon = Polygon(contour)
            if polygon.contains(support_point) or polygon.intersects(support_point):
                found = True
                break
        
        if found:
            intersection_z_values.append(z_value)

    return intersection_z_values

def visualize_slices_and_supports(slices: List[dict], supports: List[vedo.Cylinder]) -> None:
    '''
    Visualizes the slices and generated supports using vedo.
    
    :param slices: The slices as a list of dictionaries.
    :param supports: The support structures as a list of `vedo.Cylinder` objects.
    '''
    plotter = vedo.Plotter()

    for slice_data in slices:
        z_value: float = slice_data['z']
        for contour in slice_data['contours']:
            contour_points: np.ndarray = np.column_stack((contour, np.full(len(contour), z_value)))
            mesh = vedo.Line(contour_points, closed=True, lw=2)
            plotter += mesh

    for support in supports:
        plotter += support

    plotter.show()

def get_user_input(prompt: str, default: float) -> float:
    '''
    Gets user input with a default value, ensuring valid float input and providing feedback when the default is used.
    
    :param prompt: The input prompt for the user.
    :param default: The default value if no input is provided.
    :return: The input value as a float.
    '''
    while True:
        try:
            user_input = input(prompt).strip()
            if user_input == "":
                print(f"No input given! Using default: {default}")
                return default
            value = float(user_input)
            if value < 0:
                print(f"Error input '{user_input}': Must be a non-zero positive number!")
            else:
                if value > 90:
                    adjusted_value = 180 - value
                    print(f"Angle greater than 90 provided. Adjusting {value} to {adjusted_value}")
                    return adjusted_value
                return value
        except ValueError:
            print(f"Error input '{user_input}': Please enter a valid number!")



'''This function is only valid and runs when you do not run the program with poetry'''
def get_slc_file_path() -> Path:
    '''
    Prompts the user for the name of an SLC file in the `slc_samples` folder and returns its path.
    
    :return: The path to the SLC file.
    '''
    current_dir: Path = Path(__file__).parent
    slc_folder: Path = current_dir / 'slc_samples'
    file_name: str = input("Enter the name of your SLC file located in slc_samples: ")
    file_path = slc_folder / file_name
    if not file_path.exists(): 
        raise FileNotFoundError(f"The file {file_name} does not exist in the slc_samples folder.")
    return file_path


def slc_main(file_path: Path) -> None:
    '''
    Main function to handle user input, read the SLC file, and generate supports.
    '''
    builder = SupportParameterBuilder()

    spacing_between_supports = get_user_input("\n1. Please input the spacing between supports (default 0.5):  ", 0.5)
    builder.set_spacing_between_supports(spacing_between_supports)

    maximum_column_diameter = get_user_input("\n2. Please input the maximum column diameter (default 0.2): ", 0.2)
    builder.set_maximum_column_diameter(maximum_column_diameter)

    self_support_angle = get_user_input("\n3. Please input the self-support angle (default 20): ", 20)
    builder.set_self_support_angle(self_support_angle)

    spacing_from_model = get_user_input("\n4. Please input the spacing from the model (default 0.5): ", 0.5)
    builder.set_spacing_from_model(spacing_from_model)

    while True:
        contact_point_diameter = get_user_input(
            f"\n5. Please input the contact point diameter (default 0.2): ", 0.2
        )
        if contact_point_diameter <= maximum_column_diameter:
            break
        print(f"Contact point diameter ({contact_point_diameter}) cannot be greater than the maximum column diameter ({maximum_column_diameter}). Please try again.")
    
    builder.set_contact_point_diameter(contact_point_diameter)

    settings = builder.build()

    reader = SLCReader(file_path)
    slices = reader.read_slices(join_gaps=True)

    merged_polygon = merge_polygons([Polygon(contour) for slice_data in slices for contour in slice_data['contours']])

    support_generator = SupportGenerator(slices, merged_polygon, settings)
    angle_based_supports = support_generator.create_support_for_angle()
    parameter_based_supports = support_generator.create_support_for_parameters()

    all_supports = angle_based_supports + parameter_based_supports

    print(f"Total number of supports generated: {len(all_supports)}")
    visualize_slices_and_supports(slices, all_supports)

    reader.close()

if __name__ == "__main__":
    file_path = get_slc_file_path()
    slc_main(file_path)
