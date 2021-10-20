import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from silg.envs.touchdown.base_navigator import BaseNavigator

def draw_vertline(im, x, color='black', width=2, write=None):
    '''draw a vertical line in Pillow.Image object `im` at x.
    write is optional text to draw beside vertical line
    '''
    draw = ImageDraw.Draw(im)
    draw.rectangle((x, 0, x + width, im.size[1]), fill=color)
    if write is not None:
        draw.text((x + width + 3, 5), str(write), fill=color)

        
def downsample(im, ratio=5):
    ''' Downsample Pillow.Image image for faster display
    '''
    return im.resize((im.size[0] // ratio, im.size[1] // ratio))

class TDNavigator(BaseNavigator):
    '''Implmentation of the Touchdown environment with modified navigation
    API. Instead of rotating the panorama, the agent simply selects a heading
    out of a list provided in the obs.
    '''
    def __init__(self, data_json, loader_func):
        '''Parameters:
        data_json:
            Path to the provided touchdown data json
        loader_func:
            callable that will be called to load features. Expected to take as
            input a panoid (str) and return a numpy array with dim h x w x c
        '''
        
        super().__init__()
        self.loader_func = loader_func
        self.data_json = Path(data_json)
        self.dataset = [] # loaded data from self.data_json
        self.x_to_heading = {} # convert x coord to pano heading
        self.cur_id = int() # current active sample in self.dataset
        self.target_panoid = str() # panoid to try to reach
        self.graph_state = (str(), int()) # tuple of current panoid and heading (maintained by BaseNavigator)
        self.graph = self.graph # the graph used by BaseNavigator
        self._cur_obs = {} # the current observation. Stored in case we call render()
        
        with self.data_json.open(mode="r") as dj:
            for traj in dj:
                self.dataset.append(json.loads(traj))

    def step(self, action:int):
        assert action in self.x_to_heading, 'action not in valid set'
        next_heading = self.x_to_heading[action]
        
        while self.graph_state[1] != next_heading: # current heading != next heading
            super().step('right')
        super().step('forward')
        
        return self._get_obs()
        
    def reset(self, sample_id:int=None, panoid:str=None):
        if sample_id is None:
            sample_id = random.choice(range(len(self.dataset)))
        
        self.cur_id = sample_id
        self.target_panoid = self.dataset[sample_id]['main_pano']
        
        panoids = self.dataset[sample_id]['route_panoids']
        instructions = self.dataset[sample_id]['navigation_text']
        if panoid:
            self.graph_state = panoid, random.choice(list(self.graph.nodes[panoid].neighbors))
        else:
            self.graph_state = panoids[0], random.choice(list(self.graph.nodes[panoids[0]].neighbors))
        return self._get_obs(), instructions
        
    def get_cur_panoid(self):
        return self.graph_state[0]

    def heading_to_x(self, width:int, heading:int) -> int:
        # convert heading direction to x coordinate in feature of width width       
        panoid, _ = self.graph_state
        
        # compute the x coordinates of the heading
        center = width // 2
        
        shift_angle = self.graph.nodes[panoid].pano_yaw_angle - heading
        shift = int(width * shift_angle / 360)
        heading_x = (center - shift) % width
        
        # clamp the heading. This should no longer be necessary after taking modulo width
        # heading_x = max(heading_x, 0)
        # heading_x = min(heading_x, width - 1)
        
        return heading_x
        
    def _get_obs(self):
        panoid, cur_heading = self.graph_state
        neighbors = self.graph.nodes[panoid].neighbors
        features = self.loader_func(panoid)
        width = features.shape[1]

        self.x_to_heading = {}
        
        assert cur_heading in neighbors, f"heading: {cur_heading}, neighbors: {neighbors}"
        
        for heading in neighbors:
            x = self.heading_to_x(width, heading)
            self.x_to_heading[x] = heading
            if heading == cur_heading:
                cur_heading_x = x
                
        self._cur_obs = {
            'features': features,
            'x': sorted(self.x_to_heading),
            'cur_x': cur_heading_x
        }
        
        return self._cur_obs
    
    def render(self, downsample_ratio:int=1):
        im = Image.fromarray(self._cur_obs['features'])
        
        if downsample_ratio > 1:
            im = downsample(im, ratio=downsample_ratio)
            
        for i, x in enumerate(self._cur_obs['x']):
            if x == self._cur_obs['cur_x']:
                draw_vertline(
                    im=im,
                    x=self._cur_obs['cur_x'] // downsample_ratio,
                    color='red',
                    write=i
                )
            else:
                draw_vertline(
                    im=im,
                    x=x // downsample_ratio,
                    color='blue',
                    write=i
                )
        return im
    
    def close(self):
        return