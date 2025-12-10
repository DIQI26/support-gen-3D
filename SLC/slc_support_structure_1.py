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
        self.file_path: Path = file_path
        self.stream = open(file_path, 'rb')
        self.header: List[str] = []
        self.next_contour_count: int = 0
        self.next_z: float = np.nan
        self.scale: float = scale
        self.offset: np.ndarray = offset

    def device_readable(self) -> bool:
        return self.stream.readable()

    def read_header(self) -> bool:
        if not self.header:
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
        if self.device_readable() and self.next_contour_count != SLCReader.LAST_CONTOUR and self.read_header():
            return self.next_z + self.offset[2]
        else:
            return np.nan

    def read_slice(self, join_gaps: bool = False) -> Optional[dict]:
        if not self.device_readable() or self.next_contour_count == SLCReader.LAST_CONTOUR or not self.read_header():
            return None

        slice_data: dict = {
            'z': self.next_z,
            'contours': [],
            'thickness': np.nan
        }

        for _ in range(self.next_contour_count):
            vertex_count: int = int.from_bytes(self.stream.read(4), 'little')
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
        if np.isclose(self.scale, 1.0) or not vertices.size:
            return vertices
        return vertices * self.scale

    def _join_gaps(self, vertices: np.ndarray) -> np.ndarray:
        gap_free: List[np.ndarray] = []
        from_idx: int = 0

        for j in range(1, len(vertices)):
            if np.array_equal(vertices[j - 1], vertices[j]):
                gap_free.extend(vertices[from_idx:j])
                from_idx = j
        gap_free.extend(vertices[from_idx:])
        return np.array(gap_free)

    def read_slices(self, join_gaps: bool = False) -> List[dict]:
        all_slices: List[dict] = []
        while True:
            slice_data = self.read_slice(join_gaps)
            if slice_data is None:
                break
            all_slices.append(slice_data)
        return all_slices

    def close(self) -> None:
        self.stream.close()


class SupportSettings:
    def __init__(self, spacing_between_supports: float = 0.5, maximum_column_diameter: float = 0.1,
                 self_support_angle: float = 20, spacing_from_model: float = 0.5,
                 contact_point_diameter: float = 0.1):
        self.spacing_between_supports: float = spacing_between_supports
        self.maximum_column_diameter: float = maximum_column_diameter
        self.self_support_angle: float = self_support_angle
        self.spacing_from_model: float = spacing_from_model
        self.contact_point_diameter: float = contact_point_diameter
        self.support_radius: float = self.maximum_column_diameter / 2


class SupportGenerator:
    def __init__(self, slices: List[dict], merged_area: Polygon, settings: SupportSettings):
        self.slices: List[dict] = slices
        self.merged_area: Polygon = merged_area
        self.settings: SupportSettings = settings
        self.existing_support_points: List[List[float]] = []

    def create_support_for_angle(self) -> List[vedo.Cylinder]:
        supports_for_angle: List[vedo.Cylinder] = []
        max_z: float = self.slices[-1]['z']
        min_z: float = self.slices[0]['z']
        offset_area: Polygon = self.merged_area.buffer(-self.settings.spacing_from_model)

        if offset_area.is_empty:
            return supports_for_angle

        for slice_data in tqdm(self.slices, desc="Calculating angle-based supports", unit="slice"):
            for contour in slice_data['contours']:
                angle: float = calculate_angle_with_vertical(contour)
                if angle > self.settings.self_support_angle:
                    centroid: np.ndarray = np.mean(contour, axis=0)
                    intersections: List[float] = check_intersection(self.slices, centroid[0], centroid[1])

                    if intersections:
                        intersections = np.insert(intersections, 0, min_z)
                        supports_for_angle.extend(self._create_cylindrical_supports(centroid, intersections))

        return supports_for_angle

    def _create_cylindrical_supports(self, centroid: np.ndarray, intersections: List[float]) -> List[vedo.Cylinder]:
        supports: List[vedo.Cylinder] = []
        for idx, z_cor in enumerate(intersections):
            if idx % 2 == 1:
                passed_z_cor: float = intersections[idx - 1]
                height: float = z_cor - passed_z_cor - 2 * (self.settings.contact_point_diameter / 2)
                mid_z: float = (height / 2) + passed_z_cor
                if height > 0 and not supports_overlap(self.existing_support_points, centroid, self.settings.support_radius * 2):
                    support = self._create_support(centroid, mid_z, height)
                    supports.append(support)
        return supports

    def _create_support(self, centroid: np.ndarray, mid_z: float, height: float) -> vedo.Cylinder:
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
        bottom_sphere = vedo.Sphere(
            pos=(centroid[0], centroid[1], mid_z - height / 2),
            r=self.settings.contact_point_diameter / 2
        )
        support_structure = vedo.merge(cylinder, top_sphere, bottom_sphere)
        self.existing_support_points.append([centroid[0], centroid[1]])
        return support_structure

    def create_support_for_parameters(self) -> List[vedo.Cylinder]:
        supports_for_params: List[vedo.Cylinder] = []
        min_x, min_y, max_x, max_y = self.merged_area.buffer(-self.settings.spacing_from_model).bounds

        x_grid: float = self.settings.spacing_between_supports / 2
        y_grid: float = (self.settings.spacing_between_supports / 2) * math.tan(math.radians(60))

        x_coords: np.ndarray = np.arange(min_x, max_x, x_grid)
        y_coords: np.ndarray = np.arange(min_y, max_y, y_grid)

        total_iterations: int = len(x_coords) * len(y_coords)

        with tqdm(total=total_iterations, desc="Calculating parameter-based supports", unit="grid point") as pbar:
            for index, x in enumerate(x_coords):
                for y_cor in y_coords:
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


def calculate_angle_with_vertical(contour: np.ndarray) -> float:
    angles: List[float] = []
    vertical_vector: np.ndarray = np.array([0, 1])

    for i in range(len(contour) - 1):
        p1, p2 = contour[i], contour[i + 1]
        segment_vector: np.ndarray = p2 - p1
        dot_product: float = np.dot(segment_vector, vertical_vector)
        magnitude_segment: float = np.linalg.norm(segment_vector)
        magnitude_vertical: float = np.linalg.norm(vertical_vector)

        if magnitude_segment == 0 or magnitude_vertical == 0:
            continue

        cos_angle: float = dot_product / (magnitude_segment * magnitude_vertical)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle: float = np.arccos(cos_angle) * (180 / np.pi)
        angles.append(angle)

    return np.mean(angles) if angles else 0


def supports_overlap(existing_support_points: List[List[float]], new_point: List[float], min_distance: float) -> bool:
    return any(np.linalg.norm(np.array(existing_point) - np.array(new_point)) < min_distance for existing_point in existing_support_points)


def merge_polygons(polygons: List[Polygon]) -> Optional[Polygon]:
    valid_polygons: List[Polygon] = [poly for poly in polygons if poly.is_valid and not poly.is_empty]
    return unary_union(valid_polygons) if valid_polygons else None


def check_intersection(slices: List[dict], support_x: float, support_y: float) -> List[float]:
    intersection: bool = False
    support_point: Point = Point(support_x, support_y)
    intersection_z_values: List[float] = []

    for s in slices:
        z_value: float = s['z']
        contours: List[np.ndarray] = s['contours']

        found: bool = any(Polygon(contour).contains(support_point) or Polygon(contour).intersects(support_point) for contour in contours)
        
        if not intersection and found:
            intersection = True
            intersection_z_values.append(z_value)
        elif intersection and not found:
            intersection = False
            intersection_z_values.append(z_value)

    return intersection_z_values


def visualize_slices_and_supports(slices: List[dict], supports: List[vedo.Cylinder]) -> None:
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
    user_input: str = input(prompt).strip()
    return float(user_input) if user_input else default


def get_slc_file_path() -> Path:
    current_dir: Path = Path(__file__).parent
    slc_folder: Path = current_dir / 'slc_samples'
    file_name: str = input("Enter the name of your SLC file located in slc_samples: ")
    return slc_folder / file_name


def main() -> None:
    slc_file_path: Path = get_slc_file_path()
    
    settings: SupportSettings = SupportSettings(
        spacing_between_supports=get_user_input("Enter spacing between supports (default 0.5): ", 0.5),
        maximum_column_diameter=get_user_input("Enter maximum column diameter (default 0.1): ", 0.1),
        self_support_angle=get_user_input("Enter self support angle (default 20): ", 20),
        spacing_from_model=get_user_input("Enter spacing from model (default 0.5): ", 0.5),
        contact_point_diameter=get_user_input("Enter contact point diameter (default 0.1): ", 0.1)
    )

    reader: SLCReader = SLCReader(slc_file_path)
    slices: List[dict] = reader.read_slices(join_gaps=True)

    merged_polygon: Polygon = merge_polygons([Polygon(contour) for slice_data in slices for contour in slice_data['contours']])

    support_generator: SupportGenerator = SupportGenerator(slices, merged_polygon, settings)
    angle_based_supports: List[vedo.Cylinder] = support_generator.create_support_for_angle()
    parameter_based_supports: List[vedo.Cylinder] = support_generator.create_support_for_parameters()

    all_supports: List[vedo.Cylinder] = angle_based_supports + parameter_based_supports

    print(f"Total number of supports generated: {len(all_supports)}")
    visualize_slices_and_supports(slices, all_supports)

    reader.close()


if __name__ == "__main__":
    main()
