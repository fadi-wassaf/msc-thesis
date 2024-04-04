from typing import List
import pkg_resources
import sys

from tqdm.auto import tqdm

import numpy as np
import skrf as rf
from PIL import Image

from tools.circuits.interconnects import interconnect_z, ZConnect
from tools.circuits.builder.brick import Brick, innerconnect

class ChipGrid(object):
    """This class is used to interconnect a 2D rectangular grid of Brick objects with the option of having empty
    spaces in the lattice. Uses the geometric port definitions of the Brick objects to do the interconnection.

    Attributes
    ----------
    chip : rf.Network or ImpedanceLR
        Depending on what type of Brick objects are passed in, will save the corresponding interconnected model.
    """

    def __init__(self, brick_array : List[List[Brick]]) -> None:
        """With the passed in grid of Brick objects, interconnect them based on if the network contains rational 

        Parameters
        ----------
        brick_array : List[List[Brick]]
            2D List containing the brick objects that will be interconnected
        """

        self.brick_array = brick_array

        self.chip_num_r = len(self.brick_array)
        self.chip_num_c = len(self.brick_array[0])

        if self.chip_num_r == 0 or self.chip_num_c == 0:
            raise ValueError('There must be at least one Brick in the chip.')
        
        # Need to find first non-empty Brick to start the chip 'seed'
        chip = None
        first_r, first_c = None, None
        for r in range(self.chip_num_r):
            for c in range(self.chip_num_c):
                if self.brick_array[r][c]:
                    chip = self.brick_array[r][c].network.copy()
                    first_r, first_c = r, c
                    break
            if chip:
                break

        if not chip:
            raise ValueError('There must be at least one Brick in the chip.')

        # Check if all Bricks in the array are of the same type - must either use S parameters of ImpedanceLR
        brick_names = []
        for r in range(self.chip_num_r):
            for c in range(self.chip_num_c):
                if self.brick_array[r][c] is not None:
                    brick_names.append(self.brick_array[r][c].name)
                    if not isinstance(self.brick_array[r][c].network, type(chip)):
                        raise ValueError('All Bricks in the array must be of type rf.Network or ImpedanceLR. No mixing!')
        
        if len(brick_names) != len(set(brick_names)):
            raise ValueError('Make sure that all Brick objects have different names!')

        connections = 0
        num_bricks = self.chip_num_r*self.chip_num_c - sum([row.count(None) for row in self.brick_array])

        # Chip construction using S Parameters
        if isinstance(chip, rf.Network):
            pbar = tqdm(total=num_bricks, desc='Constructing chip', unit='brick(s)', ncols=125) 
            pbar.update(0)
            for r in range(self.chip_num_r):
                for c in range(self.chip_num_c):

                    # If at the first non-empty Brick, skip
                    if r == first_r and c == first_c:
                        continue

                    brick = self.brick_array[r][c]

                    # If the current Brick is empty, skip it
                    if not brick:
                        continue
                    
                    # To avoid edge cases of not added bricks to the overall chip network, we construct a composite network
                    # of the existing chip and the current brick with no connections. Afterwards, we check if the brick
                    # has any connections to the brick above it or to the brick to the left of it to make connections.
                    chip_s = chip.s
                    brick_s = brick.network.s

                    chip_n = chip_s.shape[1] # number of ports on chip so far
                    chip_n_f = chip_s.shape[0] # number of frequency points
                    brick_n = brick_s.shape[1] # number of ports on brick
                    brick_n_f = chip_s.shape[0] # number of frequency points
                    composite_n = chip_n + brick_n # number of ports on composite network

                    if chip_n_f != brick_n_f:
                        raise ValueError('Chip and brick must have the same number of frequency points.')

                    # Construct the chip and current brick composite S parameter matrix
                    composite_s = np.zeros((chip_n_f, composite_n, composite_n), dtype=complex)
                    composite_s[:, :chip_n, :chip_n] = chip_s.copy()
                    composite_s[:, chip_n:, chip_n:] = brick_s.copy()
                    new_port_names = chip.port_names + brick.network.port_names
                    chip = rf.Network(frequency=chip.frequency, s=composite_s)
                    chip.port_names = new_port_names
                    
                    # Connect top port of current brick to bottom port of the brick above it if they both exist
                    if r > 0 and brick.top_port_name():
                        brick_top = self.brick_array[r - 1][c]
                        if brick_top and brick_top.bottom_port_name():
                            p_top = brick.top_port_name()
                            p_connect = brick_top.bottom_port_name()
                            try:
                                chip = innerconnect(chip, p_connect, p_top)
                            except ValueError:
                                print('Error connecting {} to {}. Perhaps there are disjoint networks?'.format(p_connect, p_top))
                                sys.exit(1)
                            connections += 1

                    # Connect left port of the current brick to the right port of the brick to the left of it if they both exist
                    if c > 0 and brick.left_port_name():
                        brick_left = self.brick_array[r][c - 1]
                        if brick_left and brick_left.right_port_name():
                            p_left = brick.left_port_name()
                            p_connect = brick_left.right_port_name()
                            try:
                                chip = innerconnect(chip, p_connect, p_left)
                            except ValueError:
                                print('Error connecting {} to {}. Perhaps there are disjoint networks?'.format(p_connect, p_left))
                                sys.exit(1)
                            connections += 1

                    pbar.update(1)
                    pbar.refresh()

            pbar.update(1)
            pbar.close()
        
        # Chip construction using rational Z interconnection
        else:
            # Flatten the 2D list of impedances so we can easily make a list of connections between ports
            impedances = []
            brick_numbers = np.zeros((self.chip_num_r, self.chip_num_c), dtype=int)
            idx = 0
            for r in range(self.chip_num_r):
                for c in range(self.chip_num_c):
                    if self.brick_array[r][c]:
                        impedances.append(self.brick_array[r][c].network)
                        brick_numbers[r,c] = idx
                        idx += 1

            connection_list = []
            # Add connections between all the impedances and their ports
            for r in range(self.chip_num_r):
                for c in range(self.chip_num_c):
                    # If at the first non-empty Brick, skip
                    if r == first_r and c == first_c:
                        continue

                    brick = self.brick_array[r][c]
                    
                    if not self.brick_array[r][c]:
                        continue

                    if r > 0 and brick.top_port_name():
                        brick_top = self.brick_array[r - 1][c]
                        if brick_top and brick_top.bottom_port_name():
                            connection_list.append(
                                ZConnect(
                                    brick_numbers[r,c],
                                    brick.network.port_names.index(brick.top_port_name()),
                                    brick_numbers[r-1,c],
                                    brick_top.network.port_names.index(brick_top.bottom_port_name())
                                )
                            )
                            connections += 1
                            
                    if c > 0 and brick.left_port_name():
                        brick_left = self.brick_array[r][c-1]
                        if brick_left and brick_left.right_port_name():
                            connection_list.append(
                                ZConnect(
                                    brick_numbers[r,c],
                                    brick.network.port_names.index(brick.left_port_name()),
                                    brick_numbers[r,c-1],
                                    brick_left.network.port_names.index(brick_left.right_port_name())
                                )
                            )
                            connections += 1
                            
            chip = interconnect_z(impedances, connection_list)


        self.network = chip
        self.network.name = 'chip'

        # Print some information about the chip and interconnections
        print('Chip constructed with {} Bricks and {} connections.'.format(num_bricks, connections))
        print('Chip dimensions: {} rows x {} columns'.format(self.chip_num_r, self.chip_num_c))
        print('Number of ports: {}'.format(len(self.network.port_names)))
        print('Port names: {}'.format(self.network.port_names))

    def chip_image(self, scale : float = 1, label_names : bool = False, label_ports : bool = False) -> 'Image':
        """Builds and returns an image of the whole chip array.

        Parameters
        ----------
        scale : int, optional
            Scale factor for the image based on (1000x1000) pixel brick size, by default 1
        label_names : bool, optional
            Label the bricks with their names, by default False
        label_ports : bool, optional
            Label the ports with their names, by default False
            
        Returns
        -------
        Image
            Full size image of the chip array.
        """        

        # Setup grid image size and background
        brick_w = int(1000*scale)
        brick_h = int(1000*scale)
        grid_w = brick_w*self.chip_num_c
        grid_h = brick_h*self.chip_num_r
        grid_img = Image.new('RGB', (grid_w, grid_h), color=(226, 226, 226))
        
        # Load the border image in case it's used
        border = pkg_resources.resource_stream(__name__, f'./border.png')
        border_img = Image.open(border).convert('RGBA').resize((brick_w, brick_h))

        # Fill in brick images on the grid
        for r in range(self.chip_num_r):
            for c in range(self.chip_num_c):
                brick = self.brick_array[r][c]
                if brick:
                    brick_img = brick.brick_image(label_name=label_names, label_ports=label_ports).convert('RGBA')
                    brick_img = brick_img.resize((brick_w, brick_h))
                    if label_names or label_ports:
                        brick_img.paste(border_img, (0, 0), border_img)
                    grid_img.paste(brick_img, (c*brick_w, r*brick_h))
                else:
                    if label_names or label_ports:
                        grid_img.paste(border_img, (c*brick_w, r*brick_h), border_img)
        return grid_img