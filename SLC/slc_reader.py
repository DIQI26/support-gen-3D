import vedo
import numpy as np
from pathlib import Path

class slc_reader:
    '''SLC Reader class for parsing and visualizing SLC files.'''
    LAST_CONTOUR = 0xffffffff  # 32-bit maximum value
    MAXIMUM_HEADER_SIZE = 2048
    MAXIMUM_VERTEX_SIZE = 0x1fffffff

    def __init__(self, file_path):
        self.file_path = file_path
        self.stream = open(file_path, 'rb')
        self.header = []
        self.next_contour_count = 0
        self.next_z = np.nan
        self.scale = 1.0
        self.offset = np.array([0.0, 0.0, 0.0])

    def device_readable(self):
        '''Checks if the filestream is readable.'''
        return self.stream.readable()

    def read_header(self):
        '''Reads and parses the SLC file header.'''
        if not self.header:
            raw_header = bytearray()
            ch = b' '

            while True:
                line = self.stream.readline()
                if not line:
                    raw_header.clear()
                    break
                raw_header += ch + line.strip() + b' '
                ch = self.stream.read(1)
                if ch == b'\x1a':  # End of header marker
                    break

            self.header = raw_header.decode('utf-8').split()
            if not self.header or ch != b'\x1a':
                return False

            id = self.header.index('-TYPE') + 1 if '-TYPE' in self.header else -1
            if id > 0 and id < len(self.header) and self.header[id].strip() == 'WEB':
                return False

            id = self.header.index('-UNIT') + 1 if '-UNIT' in self.header else -1
            if id > 0 and id < len(self.header) and self.header[id].strip() == 'INCH':
                self.scale = 25.4

            self.stream.seek(256, 1)
            byte = int.from_bytes(self.stream.read(1), 'little')
            self.stream.seek(4 * 4 * byte, 1)
            self.next_z = np.frombuffer(self.stream.read(4), dtype=np.float32)[0]
            self.next_contour_count = int.from_bytes(self.stream.read(4), 'little')

        return True

    def next_z_value(self):
        '''Returns the z-coordinate of the next slice.'''
        if self.device_readable() and self.next_contour_count != slc_reader.LAST_CONTOUR and self.read_header():
            return self.next_z + self.offset[2]
        else:
            return np.nan

    def read_slice(self, join_gaps=False):
        '''Reads a single slice and initializes a dictionary to store slice data.'''
        if not self.device_readable() or self.next_contour_count == slc_reader.LAST_CONTOUR or not self.read_header():
            return None

        slice_data = {
            'z': self.next_z,
            'contours': [],
            'thickness': np.nan
        }

        for _ in range(self.next_contour_count):
            vertex_count = int.from_bytes(self.stream.read(4), 'little')
            gap_count = int.from_bytes(self.stream.read(4), 'little')

            if vertex_count:
                vertices = np.frombuffer(self.stream.read(min(vertex_count, slc_reader.MAXIMUM_VERTEX_SIZE) * 8), dtype=np.float32).reshape(-1, 2)

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

            adjusted_contours = []
            for contour in slice_data['contours']:
                adjusted_contour = contour + self.offset[:2]
                adjusted_contours.append(adjusted_contour)

            slice_data['contours'] = adjusted_contours

        return slice_data

    def _scale(self, vertices):
        '''Scales the vertices based on the scale factor.'''
        if np.isclose(self.scale, 1.0) or not vertices.size:
            return vertices
        return vertices * self.scale

    def _join_gaps(self, vertices):
        '''Joins gaps in the vertices.'''
        gap_free = []
        from_idx = 0

        for j in range(1, len(vertices)):
            if np.array_equal(vertices[j-1], vertices[j]):
                gap_free.extend(vertices[from_idx:j])
                from_idx = j

        gap_free.extend(vertices[from_idx:])
        return np.array(gap_free)

    def read_single_slice(self, slice_number, join_gaps=False):
        '''Reads a single specified slice.'''
        for _ in range(slice_number - 1):
            self.read_slice(join_gaps)

        slice_data = self.read_slice(join_gaps)
        return [slice_data]

    def read_slices(self, join_gaps=False):
        '''Reads all slices from the SLC file.'''
        all_slices = []
        while True:
            slice_data = self.read_slice(join_gaps)
            if slice_data is None:
                break
            all_slices.append(slice_data)
        return all_slices

    def close(self):
        '''Closes the file stream.'''
        self.stream.close()

    def visualize(self, slices, show_coordinates=False):
        '''Visualizes the slices using vedo.'''
        plotter = vedo.Plotter()

        for slice_data in slices:
            for contour in slice_data['contours']:
                contour_points = np.column_stack((contour, np.full(len(contour), slice_data['z'])))

                mesh = vedo.Line(contour_points, closed=True, lw=2)
                plotter += mesh
                points = vedo.Points(contour_points, r=5, c='red')
                plotter += points

                if show_coordinates:
                    for point in contour_points:
                        coord_text = f"({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"
                        label_pos = point + np.array([0.5, 0.5, 0])
                        label3d = vedo.Text3D(coord_text, pos=label_pos, s=0.03, c='black')
                        plotter += label3d

        plotter.show()

def main():
    '''Main entry point of the program.'''
    current_dir = Path(__file__).parent
    slc_folder = current_dir / 'slc_samples'
    file_name = input("Enter the name of your SLC file located in slc_samples: ")
    slc_file_path = slc_folder / file_name

    reader = slc_reader(slc_file_path)

    print("Choose an option:")
    print("1. Read the whole SLC file")
    print("2. Read a particular slice")
    choice = input("Enter 1 or 2: ")

    if choice == '1':
        slices = reader.read_slices()
        reader.visualize(slices)
    elif choice == '2':
        slice_number = int(input("Enter the slice number to read: "))
        slices = reader.read_single_slice(slice_number)
        reader.visualize(slices, show_coordinates=True)
    else:
        print("Invalid choice.")

    reader.close()

if __name__ == '__main__':
    main()
