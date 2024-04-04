from typing import (Tuple, Dict, List)
import pkg_resources

import copy
import skrf as rf
from PIL import Image, ImageDraw, ImageFont

from tools.circuits.impedance import ImpedanceLR

class Brick(object):
    """This class is used to represent electromagnetic brick elements that can be interconnected in a rectangular grid. 
    The bricks can have 4 "geometric" port corresponding to their 4 sides. They can also have additional ports that are
    not meant for interconnection in a grid.

    Each Brick should have a unique name so that it can be identified within the interconnection process. Port names should 
    also be specified in the order relevant to the passed in discretized S-parameter or rational impedance function.
    """

    def __init__(
        self,
        name : str,
        port_names : List[str],
        port_geo : List[str],
        orientation : int = 0,
        img_file : str = None,
        **kwargs
    ):
        """

        Parameters
        ----------
        name : str
            Unique name for this brick
        port_names : List[str]
            Names of the ports in the order corresponding to the S- or Z-parameter ports
        port_geo : List[str]
            List of port names contained in port_names that are at the sides of the rectangular brick and can be used
            for interconnection. Should be a list of 4 values. If no port is present on a side, the value should be none.
            Order of port geometry is: [Left, Top, Right, Bottom]
        orientation : int, optional
            Specifies if the geometric ports should be rotated (number of 90 degrees counter-clockwise rotations), by default 0
        img_file : str, optional
            Location of overhead image file for a given brick, by default None
        **kwargs
            Either rf_network (skrf.Network) or rational_z (ImpedanceLR) must be passed in to represent the multiport response of the brick.
        """

        self.name = name
        self.img_file = img_file

        # The following are used to keep track of the original port names and geometry (for copying)
        self.og_port_names = port_names.copy() # holds original port names
        self.og_port_geo = copy.deepcopy(port_geo) # holds original port names (but will get rotated)

        # Initialize port geometry with prefix of brick name
        # Needed later when interconnecting bricks in the grid
        self.port_geo = copy.deepcopy(self.og_port_geo)
        for i in range(len(self.port_geo)):
            if self.port_geo[i]:
                self.port_geo[i] = name + '_' + self.port_geo[i]
          
        if 'rf_network' in kwargs:
            self.network = kwargs['rf_network'].copy() # rf.Network(f = kwargs['f'], s = kwargs['s_param'])
        elif 'rational_z' in kwargs:
            self.network = kwargs['rational_z'].copy()
        else:
            raise ValueError('Must have passed in kwargs (f and s_param) or rational_z!')
        
        self.network.name = name
        self.network.port_names = [name + '_' + p for p in port_names]

        if len(port_names) != self.network.number_of_ports:
            raise ValueError('Number of port names must match number of ports in the network')
        
        self.orientation = 0
        for i in range(orientation):
            self.rotate_ports_90()

    def copy(self, name: str, orientation : int = 0) -> 'Brick':
        """Creates a copy of the Brick object using the existing network parameters,
        and original port names and geometry.

        Parameters
        ----------
        name : str
            The name of the new network - remember to make it unique
        orientation : int, optional
            The number of 90 degree counter-clockwise rotations to apply to the brick, 
            by default same as the component being copied
            
        Returns
        -------
        QBrick
            A copy of the QBrick object
        """        

        if isinstance(self.network, ImpedanceLR):
            return Brick(name, self.og_port_names, self.og_port_geo,  orientation=(orientation + self.orientation) % 4,
                    img_file=self.img_file, rational_z=self.network)
        else:
            return Brick(name, self.og_port_names, self.og_port_geo,  orientation=(orientation + self.orientation) % 4,
                    img_file=self.img_file, rf_network=self.network)

    def rotate_ports_90(self) -> None:
        """Applies a rotation of 90 degrees counter-clockwise to the port geometry of the brick.
        """
        self.orientation = (self.orientation + 1) % 4
        self.port_geo = self.port_geo[1:] + self.port_geo[:1]
        self.og_port_geo = self.og_port_geo[1:] + self.og_port_geo[:1]

    def top_port_name(self, og : bool = False) -> str:
        """Gets the name of the top port on the brick.

        Parameters
        ----------
        og : bool, optional
            Whether to use the original port names or not, by default False
        
        Returns
        -------
        str
            Port name
        """
        return self.port_geo[1] if not og else self.og_port_geo[1]
    
    def bottom_port_name(self, og : bool = False) -> str:
        """Gets the name of the bottom port on the brick.

        Parameters
        ----------
        og : bool, optional
            Whether to use the original port names or not, by default False
        
        Returns
        -------
        str
            Port name
        """
        return self.port_geo[3] if not og else self.og_port_geo[3]
    
    def right_port_name(self, og : bool = False) -> str:
        """Gets the name of the right port on the brick.

        Parameters
        ----------
        og : bool, optional
            Whether to use the original port names or not, by default False
        
        Returns
        -------
        str
            Port name
        """  
        return self.port_geo[2] if not og else self.og_port_geo[2]
    
    def left_port_name(self, og : bool = False) -> str:
        """Gets the name of the left port on the brick.

        Parameters
        ----------
        og : bool, optional
            Whether to use the original port names or not, by default False
        
        Returns
        -------
        str
            Port name
        """  
        return self.port_geo[0] if not og else self.og_port_geo[0]
    
    def brick_image(self, label_name : bool = False, label_ports : bool = False) -> 'Image':
        """Gets the image of the brick using self.img_file. If the file doesn't exist,
        it will create a blank image with the name of the brick and lines as placeholders
        for the ports. Images for bricks should be 1000x1000 pixels (or square aspect ratio).
        
        Parameters
        ----------
        label_name : bool, optional
            Label the brick name on the image, by default False
        label_ports : bool, optional
            Label the port names on the image, by default False
            
        Returns
        -------
        Image
            Image of the brick
        """  
        brick_img = None
        try:
            # Load, resize, and rotate the original brick image
            brick_img = Image.open(self.img_file)
            brick_img = brick_img.resize((1000, 1000))
            brick_img = brick_img.rotate(90*self.orientation)

            # Add the brick name to the image in the top left corner
            if label_name:
                W, H = brick_img.size
                font = ImageFont.truetype('arial.ttf', 100)
                draw = ImageDraw.Draw(brick_img)
                draw.text((20, 20), self.name, font=font, fill='black')
        except:
            # Draw a blank image with the name of the brick
            W, H = (1000, 1000)
            font = ImageFont.truetype('arial.ttf', 200)
            border = pkg_resources.resource_stream(__name__, f'./border.png')
            border_img = Image.open(border).convert('RGBA')
            img = Image.new('RGB', (1000, 1000), color=(226, 226, 226))
            img.paste(border_img, (0, 0), border_img)
            draw = ImageDraw.Draw(img)
            _, _, w, h = draw.textbbox((0, 0), self.name, font=font)
            draw.text(((W-w)/2, (H-h)/2), self.name, font=font, fill='black')

            def row_col_port_geo(i):
                return [(1,0), (0,1), (1,2), (2,1)][i]

            for i in range(len(self.port_geo)):
                if self.port_geo[i] is not None:
                    r,c = row_col_port_geo(i)
                    x1 = W/2 - (w/2) + c*(w/2)
                    y1 = H/2 - (h/2) + r*(h/2)
                    x2 = c*W/2
                    y2 = r*H/2
                    draw.line((x1, y1, x2, y2), fill='black', width=10)


            brick_img = img
            
        # Label the ports
        if label_ports:
            font = ImageFont.truetype('arial.ttf', 50)
            draw = ImageDraw.Draw(brick_img)
            W,H = brick_img.size

            # Top port
            if self.top_port_name(og=True):
                _, _, w, h = draw.textbbox((0, 0), f'.{self.top_port_name(og=True)}.', font=font)
                draw.text((W/2 + w/4, h/4), self.top_port_name(og=True), font=font, fill='black')
            # Bottom port
            if self.bottom_port_name(og=True):
                _, _, w, h = draw.textbbox((0, 0), f'.{self.bottom_port_name(og=True)}.', font=font)
                draw.text((W/2 + w/4, H - 5*h/4), self.bottom_port_name(og=True), font=font, fill='black')
            # Right port
            if self.right_port_name(og=True):
                _, _, w, h = draw.textbbox((0, 0), f'.{self.right_port_name(og=True)}.', font=font)
                draw.text((W - w + w/8, H/2 + h/2), self.right_port_name(og=True), font=font, fill='black')
            # Left port
            if self.left_port_name(og=True):
                _, _, w, h = draw.textbbox((0, 0), f'.{self.left_port_name(og=True)}.', font=font)
                draw.text((w/8, H/2 + h/2), self.left_port_name(og=True), font=font, fill='black')
        
        return brick_img
    
def innerconnect(A: rf.Network, p1_name: str, p2_name: str) -> rf.Network:
    """Innerconnects two ports on the same network using the string names of the ports.
    Keeps track of the port names and removes the old ones.
    Used for S-parameter sub-network growth in grid builder.

    Parameters
    ----------
    A : rf.Network
        the network with the two ports you want to connect
    p1_name : str
        first port name
    p2_name : str
        seocnd port name

    Returns
    -------
    rf.Network
        the new network with the two ports connected
    """    
    p1 = A.port_names.index(p1_name)
    p2 = A.port_names.index(p2_name)
    B = rf.network.innerconnect(A, p1, p2)
    B.port_names.remove(p1_name)
    B.port_names.remove(p2_name)
    return B