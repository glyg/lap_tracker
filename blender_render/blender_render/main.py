import sys
import os
import random

import bpy
import bmesh

import pandas as pd
import numpy as np

from blender_render import utils
from  blender_render import objects



def main():

    frame_rate = 30
    speed = 10
    color = '#2887c8'
    source_inplace = os.path.join(os.path.dirname(bpy.data.filepath), "track.h5")
    fname = os.path.join(source_inplace)
    
    scene = utils.clear_scene()
    scene.render.fps = frame_rate
    data_store = pd.HDFStore(fname)
    data = data_store['sorted']

    data = data[['x', 'y', 'z']]

    data['x'] -= data['x'].mean()
    data['y'] -= data['y'].mean()
    data['z'] -= data['z'].mean()
    
    labels = np.unique(data.index.get_level_values(1).values)
    particles = {}
    for label in labels:

        segment = data.xs(label, level=1)[['x', 'y', 'z']]
        timelapse = segment.index

        particles[label] = objects.set_moving_sphere('particle_%s' % label,
                                                     segment['x'].values,
                                                     segment['y'].values,
                                                     segment['z'].values,
                                                     color, timelapse,
                                                     frame_rate, speed, radius=2.)
    scene.frame_current = 0
    scene.frame_start = 0
    scene.frame_end = max(timelapse) * (frame_rate / speed)

    # # Smooth objects view
    # utils.select([p for p in particles.values()])
    # bpy.ops.object.shade_smooth()

    # Add an empty object and move it to the center of kts and spbs
    bpy.ops.object.empty_add()
    center = bpy.context.object
    center.name = "center_of_scene"
    center.hide = True
    timelapse = data.index.get_level_values(0)
    for i, t in enumerate(timelapse):
        x, y, z = data.loc[t][['x', 'y', 'z']].mean(axis=0)
        center.location = (x, y, z)
        center.keyframe_insert('location', frame=t * (frame_rate / speed))

    # Add a camera
    bpy.ops.object.camera_add(view_align=True, enter_editmode=False)
    camera = bpy.context.object
    camera.name = "camera"

    # Constraint camera to track center object
    bpy.context.scene.objects.active = center
    utils.select([center, camera])
    bpy.ops.object.track_set(type='TRACKTO')
    utils.unselect_all()

    scene.objects.active = None
