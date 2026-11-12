#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic herbarium dataset generation script
@author: Caleb Powell
"""
import bpy
import bmesh
from mathutils.bvhtree import BVHTree
from mathutils import Matrix, Quaternion, Vector
from random import seed as rand_seed
from random import uniform, randint, choice, choices, sample
from random import random as _random
from math import radians, prod, pi, degrees, sqrt, cos, sin
from statistics import median as stats_median
import time

import sys
import importlib
from addon_utils import check, paths, enable

import time
import os
from glob import glob

import json  #necessary for sample generation detail files.

# class IDs are intentionally spaced out such that they can be visibly differentiated
class_ids = {'other':0,
             'paper':5,
             'crc':50,
             'scale':100,
             'bark':150,
             'label':200,
             'leaf':250}

def reset_scene():

    try:
        bpy.ops.object.select_all(action='DESELECT')
    except RuntimeError:
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
    try:  # remove all potential specimen objects
        bpy.ops.object.select_pattern(pattern='specimen*')
        bpy.ops.object.select_pattern(pattern='Tree*')
        bpy.ops.object.select_pattern(pattern='Bark*')
        bpy.ops.object.select_pattern(pattern='Leaf*')
        bpy.ops.object.select_pattern(pattern='washer_dup_*')
        try:
            bpy.data.objects['label'].select_set(True)
        except KeyError:
            pass

        purge_data = set(o.data for o in bpy.context.selected_objects if o.data)
        bpy.data.batch_remove(bpy.context.selected_objects)
        bpy.data.batch_remove([o for o in purge_data if not o.users])
        bpy.ops.outliner.orphans_purge(do_recursive=True)

    except KeyError:
        print("keyerror at reset")
        pass

def reset_set():
    """
    resets set pieces to default settings, useful for leaving the set tidy.
    """
    paper_material = bpy.data.materials["Paper 2"]
    paper_material.node_tree.nodes["Mix"].inputs[0].default_value = 0.5

    herb_paper = bpy.data.objects['herb_paper']
    herb_paper.location = (0, 0, 0.0075)
    herb_paper.rotation_euler.z = 0
    herb_paper.active_material = paper_material

    light_box = bpy.data.objects['light_box']
    light_box.active_material = bpy.data.materials['Diffused_Plastic']

    black_mat = bpy.data.objects['black_mat']
    black_mat.active_material = bpy.data.materials['Black_matte']

    for light in bpy.data.collections['lighting_array'].objects:
        light.data.color = (1.0, 1.0, 1.0)
        light.data.energy = 10
        light.hide_render = False

    bpy.data.objects['Camera'].location = (0, 0, 2.1605)
    
    # make sure misc props are reset
    for prop_name in ['crc_nano', 'crc_tiffen',
                      'crc_xrite', 'crc_cameratrax',
                      'scale_1', 'scale_2',
                      'washer_1', 'washer_2']:
        prop = bpy.data.objects[prop_name]
        # consistent naming conventions allow this to happen at same iteration
        prop.hide_render = True
        prop.hide_viewport = True
        prop.scale = (1,1,1)  # still need to reset the scale
    bpy.context.view_layer.update() 

def randomize_set():
    """
    Makes random, and slight modifications to static set pieces
    """
    # wiggle the herb_paper's z rotation
    herb_paper = bpy.data.objects['herb_paper']
    herb_paper.rotation_euler.z = uniform(-0.0185, 0.0185)
    herb_paper.location.x += uniform(-0.015, 0.015)
    herb_paper.location.y += uniform(-0.015, 0.015)
    # adjust the paper's degree of 'yellowing'
    if randint(1,3) == 3:
        # 1/3rd of the time it gets a crisp new sheet
        bpy.data.materials["Paper 2"].node_tree.nodes["Mix"].inputs[0].default_value = 1
    else:  # 2/3rds of the time it ... has some degree of yellowing
        bpy.data.materials["Paper 2"].node_tree.nodes["Mix"].inputs[0].default_value = uniform(0.05, 0.95)
        bpy.data.materials["Paper 2"].node_tree.nodes["ColorRamp"].inputs[0].default_value = uniform(0.0, 1.0)
    # wiggle the camera's distance
    bpy.data.objects['Camera'].location.z = uniform(1.99, 2.45)
    # wiggle the lighting array
    for light in bpy.data.collections['lighting_array'].all_objects:
        light.data.color = (uniform(0.7, 1),
                                        uniform(0.7, 1),
                                        uniform(0.7, 1))
        light.data.energy = uniform(11, 24)
    
    # randomly locate or hide various prop pieces
    # first establish a useful reference for prop placement 
    max_x = herb_paper.location.x + (herb_paper.dimensions.x / 2)
    min_x = herb_paper.location.x - (herb_paper.dimensions.x / 2)
    max_y = herb_paper.location.y + (herb_paper.dimensions.y / 2)
    min_y = herb_paper.location.y - (herb_paper.dimensions.y / 2)

    washer_qty = 0  # set a default value, which *might* be overwritten.
    # randomly select if washers should be placed
    if randint(1, 5) == 3:
        washer_1 = bpy.data.objects['washer_1']
        washer_2 = bpy.data.objects['washer_2']
        washer_qty = randint(1, 6)
        for i in range(washer_qty):
            bpy.ops.object.select_all(action='DESELECT')
            # pick a washer to duplicate, washer type 2 is more common
            source = choice([washer_2, washer_2, washer_2, washer_1])
            washer_data = source.data.copy()
            washer_copy = bpy.data.objects.new(f"washer_dup_{i}", washer_data)
            bpy.context.collection.objects.link(washer_copy)
            scale_val = uniform(0.75, 1.25)  # randomly scale the washer
            washer_copy.scale = (scale_val, scale_val, scale_val)
            washer_copy.location.x = uniform(-0.458265, 0.458265)  #randomly ransform the washer
            washer_copy.location.y = uniform(-0.354818, 0.354818)
            washer_copy.location.z = 0.009
            washer_copy.hide_render = False
            washer_copy.hide_viewport = False

    # randomly select which crc should be used.. if any
    crc_type = choice([False, "crc_tiffen", 
                       "crc_nano", "crc_xrite",
                       "crc_cameratrax"])
    
    crc_scale = False
    if crc_type:
        prop = bpy.data.objects[crc_type]  # get a handle on the chosen crc
        prop.hide_render = False
        prop.hide_viewport = False
        
        # feed the crc material a random value, to determine if randomlized noise should be applied
        bpy.data.materials[crc_type].node_tree.nodes['Value'].outputs[0].default_value = uniform(0.0, 1.0)
        
        if crc_type == 'crc_tiffen':
            crc_scale = uniform(0.95, 1.05)  # initial scale 1 should be accurate but looks too small...
            prop.scale = (crc_scale , crc_scale , 1)
            # This magic number (0.0617...) is half the asset's x dimension)
            #  so, values over that number pull it past the paper's edge, lower values pull it toward center
            prop_offset = 0.0617249943316 + uniform(-0.1, 0.04)
            # 1,2 positions start at y+ position
            # hardcoded values were manually derived within scene
            position = randint(1,2)
            if position == 1:
                prop.rotation_euler.z = uniform(-0.00872665, 0.00872665)
                prop.location.x = uniform(-0.08, 0.08)
                prop.location.y = max_y + prop_offset
            elif position == 2:
                prop.rotation_euler.z = uniform(3.13287, 3.15032)
                prop.location.x = uniform(-0.08, 0.08)
                prop.location.y = min_y - prop_offset
            
        elif crc_type == 'crc_nano':
            crc_scale = uniform(0.95, 2.5)  # initial scale 1 should be accurate but looks too small...
            prop.scale = (crc_scale, crc_scale, 1)
            # wiggle rotation a little
            prop.rotation_euler.z = choice([uniform(1.55334, 1.58825),
                                            uniform(3.12414, 3.15905),
                                            uniform(-1.58825, -1.55334),
                                            uniform(-3.15905, -3.12414)])
            prop_offset = ((prop.dimensions.x + prop.dimensions.y) / 2) + uniform(-0.01, 0.01)
            # 1,2,3,4 positions start at y+ position and incriment clockwise
            # hardcoded values were manually derived within scene
            position = randint(1,4)
            if position == 1:
                prop.location.x = uniform(-0.46, 0.46)
                prop.location.y = max_y + prop_offset
            elif position == 2:
                prop.location.x = max_x + prop_offset
                prop.location.y = uniform(-0.31, 0.31)
            elif position == 3:
                prop.location.x = uniform(-0.46, 0.46)
                prop.location.y = min_y - prop_offset
            elif position == 4:
                prop.location.x = min_x - prop_offset
                prop.location.y = uniform(-0.31, 0.31)

        elif crc_type == 'crc_xrite':
            crc_scale= uniform(0.85, 1.15)
            prop.scale = (crc_scale, crc_scale, 1)
            # wiggle rotation a little
            prop.rotation_euler.z = choice([uniform(1.55334, 1.58825),
                                            uniform(3.12414, 3.15905),
                                            uniform(-1.58825, -1.55334),
                                            uniform(-3.15905, -3.12414)])
            prop_offset = ((prop.dimensions.x + prop.dimensions.y) / 4) + uniform(-0.04, 0.05)
            # 1,2,3,4 positions start at y+ position and incriment clockwise
            # hardcoded values were manually derived within scene
            position = randint(1,4)
            if position == 1:
                prop.location.x = uniform(-0.4, 0.4)
                prop.location.y = max_y - prop_offset
            elif position == 2:
                prop.location.x = max_x - prop_offset
                prop.location.y = uniform(-0.28, 0.28)
            elif position == 3:
                prop.location.x = uniform(-0.4, 0.4)
                prop.location.y = min_y + prop_offset
            elif position == 4:
                prop.location.x = min_x + prop_offset
                prop.location.y = uniform(-0.28, 0.28)
        
        elif crc_type == 'crc_cameratrax':
            crc_scale = uniform(0.9, 1.15)
            prop.scale = (crc_scale, crc_scale, 1)
            prop_offset = prop.dimensions.y/2 + uniform(-0.06, 0.035)
            # 1,2,3,4 positions start at y+ position and incriment clockwise
            # hardcoded values were manually derived within set
            position = randint(1,4)
            if position == 1:
                # wiggle rotation a little
                prop.rotation_euler.z = choice([uniform(3.12414, 3.15905),
                                                uniform(-3.15905, -3.12414)])
                prop.location.x = uniform(-0.385, 0.385)
                prop.location.y = max_y + prop_offset
            elif position == 2:
                prop.rotation_euler.z = choice([uniform(1.55334, 1.58825),
                                                uniform(-1.58825, -1.55334)])
                prop.location.x = max_x + prop_offset
                prop.location.y = uniform(-0.28, 0.28)
            elif position == 3:
                prop.rotation_euler.z = choice([uniform(3.12414, 3.15905),
                                                uniform(-3.15905, -3.12414)])
                prop.location.x = uniform(-0.385, 0.385)
                prop.location.y = min_y - prop_offset
            elif position == 4:
                prop.rotation_euler.z = choice([uniform(1.55334, 1.58825),
                                                uniform(-1.58825, -1.55334)])
                prop.location.x = min_x - prop_offset
                prop.location.y = uniform(-0.28, 0.28)

    # handle scale placement
    scale_name = False  # set a default value in case no scalebar is used
    if randint(1,3) < 3:  # 1/3rd of the time there is no scalebar
        scale_name = choice(['scale_1', 'scale_2'])
        prop = bpy.data.objects[scale_name ]
        prop.hide_render = False
        prop.hide_viewport = False
        # wiggle rotation a little

        if scale_name  == 'scale_1':
            # based on static dimensions
            prop_offset = 0.0112387500703 + uniform(-0.005, 0.005)
        else:
            prop_offset = 0.0149849997833 + uniform(-0.005, 0.005)
        # 1,2,3,4 positions start at y+ position and incriment clockwise
        # hardcoded values were manually derived within scene
        position = randint(1,4)
        if position == 1:
            prop.rotation_euler.z = choice([uniform(1.55334, 1.58825),
                                            uniform(-1.58825, -1.55334)])
            prop.location.x = uniform(-0.36, 0.36)
            prop.location.y = max_y + prop_offset
        elif position == 2:
            prop.rotation_euler.z = choice([uniform(3.12414, 3.15905),
                                            uniform(-3.15905, -3.12414)])
            prop.location.x = max_x + prop_offset
            prop.location.y = uniform(-0.27, 0.27)
        elif position == 3:
            prop.rotation_euler.z = choice([uniform(1.55334, 1.58825),
                                            uniform(-1.58825, -1.55334)])
            prop.location.x = uniform(-0.36, 0.36)
            prop.location.y = min_y - prop_offset
        elif position == 4:
            prop.rotation_euler.z = choice([uniform(3.12414, 3.15905),
                                            uniform(-3.15905, -3.12414)])
            prop.location.x = min_x - prop_offset
            prop.location.y = uniform(-0.27, 0.27)

    bpy.context.view_layer.update()
    label_fn = setup_label()
    set_params = { "crc_scale":crc_scale,
                  "crc_type" : crc_type,
                  "label_fn":label_fn,
                  "scale_name":scale_name,
                  "washer_qty":washer_qty,
                  }

    return set_params

def setup_label():
    herb_paper = bpy.data.objects['herb_paper']
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = None    
    
    x_scale = uniform(1.3, 1.7)
    y_scale = uniform(0.8, 1.2)
    bpy.ops.mesh.primitive_plane_add(size=0.1, location=(0,0,0.00085),
                                        scale=(x_scale, y_scale, 0))
    label = bpy.context.object
    # scale everything up or down a little
    label.scale.x = uniform(1.3, 1.7)
    label.scale.y = uniform(0.8, 1.2)
    label.name = 'label'
    
    # set the material, and load a random texture from the pregenerated labels
    label_mat_name = choice(['label_material','label_material_alt'])
    label_mat = bpy.data.materials.get(label_mat_name)
    label.data.materials.append(label_mat)
    fn = f"{randint(0, 9999)}.jpg"
    path = f".//label_generation/output/{fn}"
    bpy.data.images.load(path)
    label_mat.node_tree.nodes["Image Texture"].image = bpy.data.images[fn]    

    # give the plane some thickness
    solid_mod = label.modifiers.new('solidify', 'SOLIDIFY')
    solid_mod.thickness = uniform(0.00015, 0.00040)  # set to 0.1mm
    bpy.ops.object.modifier_apply(modifier='solidify')

    label.rotation_euler.z = choice([uniform(-0.0349, 0.0349),
                                     uniform(1.536, 1.606),
                                     uniform(-1.606, -1.536)])
    
    # finalize the current rotation and scale
    bpy.ops.object.transform_apply(rotation=True, scale=True)

    if randint(1, 2) == 1: # frequently goof with the scale
        label.scale = (uniform(0.75, 2), uniform(0.75, 1.5), 1)
    
    # do it again, so the next steps are accurate
    bpy.ops.object.transform_apply(rotation=True, scale=True)
    
    # snap the label to the herb paper
    shrink_mod = label.modifiers.new('shrinkwrap', 'SHRINKWRAP')
    shrink_mod.target = herb_paper
    shrink_mod.wrap_mode = 'OUTSIDE_SURFACE'
    shrink_mod.offset = uniform(0.15, 0.33)  # make sure it is outside

    # copy the herb_paper's z rotation
    label.rotation_euler.z = herb_paper.rotation_euler.z

    ref_max_x = herb_paper.location.x + herb_paper.dimensions.x / 2
    ref_min_x = herb_paper.location.x - herb_paper.dimensions.x / 2
    ref_max_y = herb_paper.location.y + herb_paper.dimensions.y / 2
    ref_min_y = herb_paper.location.y - herb_paper.dimensions.y / 2
    # pick a quadrant to affix the label
    # typiclly it would be "bottom right" 
    # but I want networks to learn an even distribution of locations
    quadrant = choice([1,2,3,4]) # quadrant presumes z-facing view with +y up
    padding_x = uniform(0.0005, 0.0045)
    padding_y = uniform(0.0005, 0.0045)
    x_offset = label.location.x + label.dimensions.x / 2 + padding_x
    y_offset = label.location.y + label.dimensions.y / 2 + padding_y
    if quadrant == 1:
        label.location.x = ref_max_x - x_offset
        label.location.y = ref_max_y - y_offset
    elif quadrant == 2:
        label.location.x = ref_max_x - x_offset
        label.location.y = ref_min_y + y_offset
    elif quadrant == 3:
        label.location.x = ref_min_x + x_offset
        label.location.y = ref_min_y + y_offset
    elif quadrant == 4:
        label.location.x = ref_min_x + x_offset
        label.location.y = ref_max_y - y_offset
    
    # set the object's pass Index based on the class_ids dict
    #  This is used for the annotation map generation
    label.pass_index = class_ids.get("label", 0)
    
    return fn  # pass back the filename for record keeping

def setup_addon():
    tg = 'tree-gen'  #addon name https://github.com/friggog/tree-gen
    tg = enable(tg)  # enable and return module to tg var
    assert (tg != None)  # ensure tg object was properly enabled

    #### This block re-loads all various preset tree parameters
    param_elements = tg.gui._get_tree_types()
    tree_types = {}
    for element in param_elements:
        modpath=element[0]
        # Delete old module if it exists, forcing Python to load the new file
        if modpath in sys.modules:
            del sys.modules[modpath]
        importlib.invalidate_caches()  # Make sure Python doesn't import from __pycache__
        mod = importlib.import_module(modpath)
        tree_types[element[-1]] = mod.params # organize a dict of tree type options by name:params

    # Monkey patch the addon, creating a leaf mesh function to properly set individual origins
    #   This was necessary to add a leaf specific constraint resulting in many of them facing Y-
    #   which more or less approximates the process of orienting the specimen before pressing it.
    Vector = tg.chturtle.Vector
    windman = bpy.context.window_manager
    Leaf = tg.leaf.Leaf
    Stem = tg.parametric.gen.Stem  # get a handle on a class used in modded_points_for_floor_split
    rand_in_range = tg.parametric.gen.rand_in_range
 
    def modded_create_leaf_mesh(self):
        """
            heavily modified from tree-gen's function.
            overriding was necessary to influence individual leaf positions.
                https://github.com/friggog/tree-gen
        """

        if len(self.leaves_array) <= 0:
            return

        print('\nMaking Leaves\n')
        start_time = time.time()

        # Start loading spinner
        windman.progress_begin(0, len(self.leaves_array))

        # go through global leaf array populated in branch making phase and add polygons to mesh
        base_leaf_shape = Leaf.get_shape(self.param.leaf_shape, self.tree_scale / self.param.g_scale,
                                         self.param.leaf_scale, self.param.leaf_scale_x)

        base_blossom_shape = Leaf.get_shape(-self.param.blossom_shape, self.tree_scale / self.param.g_scale,
                                            self.param.blossom_scale, 1)
        leaf_verts = []
        leaf_faces = []
        leaf_obj_data = []
        leaf_index = 0
        blossom_verts = []
        blossom_faces = []
        blossom_index = 0

        counter = 0
        for leaf in self.leaves_array:
            # Update loading spinner periodically
            if counter % 500 == 0:
                windman.progress_update(counter / 100)
                print('\r-> {} leaves made, {} blossoms made'.format(leaf_index, blossom_index))
            if self.param.blossom_rate and _random() < self.param.blossom_rate:
                verts, faces = leaf.get_mesh(self.param.leaf_bend, base_blossom_shape, blossom_index)
                blossom_verts.extend(verts)
                blossom_faces.extend(faces)
                blossom_index += 1
            else:
                verts, faces = leaf.get_mesh(self.param.leaf_bend, base_leaf_shape, leaf_index)
                leaf_obj_data.append((verts, leaf))
                leaf_index += 1
            counter += 1

        # set up mesh object
        if leaf_index > 0:

            leaves = []
            for verts, leaf in leaf_obj_data:
                position = leaf.position
                faces = base_leaf_shape[1]
                leaf = bpy.data.meshes.new('leaf')
                leaf.from_pydata(verts, (), faces)
                leaf_obj = bpy.data.objects.new("Leaf", leaf)
                bpy.context.collection.objects.link(leaf_obj)
                leaf_obj.parent = self.tree_obj

                # origin relocation code adapted from batFINGER
                #     see: https://blender.stackexchange.com/questions/35825/changing-object-origin-to-arbitrary-point-without-origin-set
                mw = leaf_obj.matrix_world
                imw = mw.inverted()
                me = leaf_obj.data
                origin = position
                local_origin = imw @ origin
                me.transform(Matrix.Translation(-local_origin))
                mw.translation += (origin - mw.translation)

                leaf_uv = base_leaf_shape[2]
                if leaf_uv:
                    leaf.uv_layers.new(name="leavesUV")
                    uv_layer = leaf.uv_layers.active.data

                    for seg_ind in range(int(len(faces) / len(base_leaf_shape[1]))):
                        vert_ind = 0
                        for vert in leaf_uv:
                            uv_layer[seg_ind * len(leaf_uv) + vert_ind].uv = vert
                            vert_ind += 1
                            

                leaves.append((leaf_obj, position))
            # now constrain and join them
            position_leaves(leaves)
        if blossom_index > 0:
            blossom = bpy.data.meshes.new('blossom')
            blossom_obj = bpy.data.objects.new('Blossom', blossom)
            bpy.context.collection.objects.link(blossom_obj)
            blossom_obj.parent = self.tree_obj
            blossom.from_pydata(blossom_verts, (), blossom_faces)

        l_time = time.time() - start_time
        print('\nMade %i leaves and %i blossoms in %f seconds\n' % (leaf_index, blossom_index, l_time))
        windman.progress_end()

    def modded_points_for_floor_split(self):
        """
        overriding this function from addon is necessary to break occasional endless loops
        caused by insane random param values.
        Calculate Poissonly distributed points for stem start points
        """
        array = []

        # calculate approx spacing radius for dummy stem
        self.tree_scale = self.param.g_scale + self.param.g_scale_v
        stem = Stem(0, None)
        stem.length = self.calc_stem_length(stem)
        rad = 2.5 * self.calc_stem_radius(stem)

        # generate points
        for _ in range(self.param.branches[0]):
            point_ok = False
            i = 0
            while not point_ok:
                i += 1
                # distance from center proportional for number of splits, tree scale and stem radius
                dis = sqrt(_random() * self.param.branches[0] / 2.5 * self.param.g_scale * self.param.ratio)
                # angle random in circle
                theta = rand_in_range(0, 2 * pi)
                pos = Vector([dis * cos(theta), dis * sin(theta), 0])
                # test point against those already in array to ensure it will not intersect
                point_m_ok = True
                for point in array:
                    if (point[0] - pos).magnitude < rad:
                        point_m_ok = False
                        break
                if point_m_ok:
                    point_ok = True
                    array.append((pos, theta))
                elif i > 50:
                    raise ValueError
        return array

    tg.parametric.gen.Tree.create_leaf_mesh = modded_create_leaf_mesh
    tg.parametric.gen.Tree.points_for_floor_split = modded_points_for_floor_split
    
    # Set of bound ranges used for tree-gen generation.
    param_ranges = {'leaf_blos_num': (-2, 50),
                    'branches': [(1, 50), (1, 32), (0, 120), (0, 300)],
                    'leaf_scale': (0.85, 4.0),
                    'leaf_scale_x': (0.7, 2.0),
                    'g_scale': (20, 25),
                    'g_scale_v': (0, 3),
                    'ratio_power': (1, 2),
                    'tropism': [(0, 0), (0, 0), (0, 3)],
                    'radius_mod': [(1, 1), (1, 1), (0.5, 1), (1, 1)],
                    'shape': (0, 7),
                    'length': [(0.8, 1), (0.2, 1), (0.3, 2.0), (0.1, 0.7)],
                    'length_v': [(0, 0.2), (0, 0.1), (0, 0.2), (0, 0)],
                    'down_angle': [(0, 0), (20, 100), (-10, 70), (0, 45)],
                    'down_angle_v': [(0, 0), (-80, 20), (-75, 30), (0, 30)],
                    # note leaf_shape is handled manually, omitted here to avoid automatic creation
                    #'leaf_shape': (0, 10),
                    'base_size': [(0.01, 0.95), (0, 0.4), (0, 0.05), (0, 0.05)],
                    'levels': (2, 4),
                    'curve': [(0, 50), (-60, 50), (-60, 0), (0, 0)],
                    'curve_v': [(-40, 200), (0, 400), (-30, 500), (0, 100)],
                    'curve_res': [(1, 20), (3, 17), (0, 12), (0, 3)],
                    'blossom_rate': (0.35, 1),
                    # note seg_splits handled manually, omitted here to avoid automatic creation
                    #'seg_splits': [(0, 1.5), (0, 2), (0, 0.2), (0, 0)],
                    'ratio': (0.005, 0.04),
                    'leaf_bend': (0, 0.7),
                    'taper': [(0.95, 2.15), (1, 1.4), (0, 1), (0, 1)],
                    'blossom_shape': (1, 1),
                    # I believe this is a legacy param, it throws an error when present
                    #'leaves': (15, 30),
                    'branch_dist': [(0, 0), (0, 3.5), (0, 0), (0, 0)],
                    'base_splits': (-2, 2),
                    'bend_v': [(0, 150), (0, 160), (0, 250), (0, 0)],
                    'rotate': [(0, 0), (-120, 160), (-125, 260), (-90, 140)],
                    'rotate_v': [(0, 0), (0, 60), (0, 50), (0, 30)],                    
                    'curve_back': [(-40, 25), (-100, 80), (0, 0), (0, 0)],
                    'split_angle': [(0, 50), (-80, 50), (0, 50), (0, 0)],
                    'split_angle_v': [(0, 20), (-30, 10), (0, 20), (0, 0)],
                    'flare': (0, 1.2),
                    'blossom_scale': (0.08, 0.1)}


    return tg, tree_types, param_ranges

def check_y_orient(obj):
    """
    code modified from batFINGER's
        see: https://blender.stackexchange.com/questions/96066/how-to-get-all-faces-based-on-a-certain-angle-of-their-normals#answer-96121
    """
    
    me = obj.data
    forward = Vector((0, 1, 0)) #  y axis.
    backward = Vector((0, -1, 0)) #  -y axis.
    #test_angle = radians(randint(45, 65))
    test_angle = radians(randint(50, 60))
    f = me.polygons[0]
    if (f.normal.angle(forward) < test_angle) | (f.normal.angle(backward) < test_angle):
        return True
    else:
        return False


def position_leaves(leaves):
    """
    called during leaf mesh generation. Attempts to orient the leaf faces such that
    their normals either point toward or away from a reference object offset on y axis.
    This is akin to orienting the leaves under blotter paper in a press.
    """
    if leaves:
        ref = bpy.data.objects['ref_cube']
        # set up 3 options for ratios of leaf facing / away
        facing_options = choice([['TRACK_NEGATIVE_Y', 'TRACK_Y'],
                                    ['TRACK_NEGATIVE_Y', 'TRACK_Y', 'TRACK_Y'],
                                    ['TRACK_Y', 'TRACK_NEGATIVE_Y', 'TRACK_NEGATIVE_Y']])

        bpy.ops.object.select_all(action='DESELECT')
        for leaf, leaf_position in leaves:
            # orient the reference cube such that only y is aligned with constraint
            ref.location.x = leaf_position.x
            ref.location.z = leaf_position.z
            leaf.select_set(True)
            bpy.context.view_layer.objects.active = leaf
            constraint = leaf.constraints.new(type='DAMPED_TRACK')
            constraint.track_axis = choice(facing_options)
            constraint.target = ref
            constraint.influence = uniform(0.85, 1.0)

            mw = leaf.matrix_world.copy()
            constraint.influence = 0
            leaf.matrix_world = mw
            leaf.select_set(False)

        bpy.context.view_layer.update()
        bpy.ops.object.select_all(action='DESELECT')
        # I think a layer update is required, so split this process up into two loops to avoid
        # calling it every iteration of first.
        keep_leaves = []
        for leaf, _ in leaves:
            # test the object's y rotation vs global y
            # anything too skewed will be 'pressed' down to a thin line.. so remove it.
            if not check_y_orient(leaf):
                leaf.select_set(True)
            else:
                keep_leaves.append(leaf)
        bpy.data.batch_remove(bpy.context.selected_objects)

        if keep_leaves:
            first_leaf = keep_leaves[0]
            bpy.context.view_layer.objects.active = first_leaf
            for leaf in keep_leaves:
                leaf.select_set(True)
            bpy.ops.object.join()
            first_leaf.name = "Leaves"  # set the name back to expected conditions
        bpy.ops.object.select_all(action='DESELECT')

def enforce_complexity_ceil(params, b_ceil=75, l_ceil=200):
    """
    random param generation needs policing to avoid absurd values
    """
    levels = params.get('levels', 3)
    splits = params.get('seg_splits')
    splits = [x + 1 if x < 1 else x for x in splits]
    under_ceil = False
    while not under_ceil:
        branches = params.get('branches', [1, 30, 10, 10])
        b = [branches[i] * splits[i] for i in range(4)]  # multiply each branch number by propensity to split
        est_total_branches = prod(b[:levels])
        under_ceil = b_ceil > est_total_branches
        if not under_ceil:
            max_b_loc = branches.index(max(branches[:levels]))
            branches[max_b_loc] = max([0, branches[max_b_loc]-2])
            params['branches'] = branches

    under_ceil = False
    leaf_blos_num = params.get('leaf_blos_num', 40)
    while not under_ceil:
        
        est_total_leaves = leaf_blos_num * est_total_branches
        under_ceil = l_ceil > est_total_leaves
        if not under_ceil:
            leaf_blos_num -= 1

    print("est branches:", est_total_branches)
    print("est leaves:", est_total_leaves)

    params['leaf_blos_num'] = leaf_blos_num

    return params

def get_sp_params(addon_info):
    """
    Used to assign specimen specific params, not related to tree-gen's params.
    """
        # hardcoded list for ease of iterative editing. 
        #  removed from tree-gen list: 'Weeping Willow O', 'Sphere Tree'
    tg, tree_types, param_ranges = addon_info

    tree_candidates = ['Acer', 'Apple', 'Silver Birch', 'Balsam Fir',
                        'Black Oak', 'Hill Cherry', 'Cambridge Oak', 'Lombardy Poplar',
                        'Quaking Aspen', 'Douglas Fir', 'Sassafras',
                        'Weeping Willow', 'Black Tupelo',
                        'Small Pine', 'European Larch']

    selected_tree = choice(tree_candidates)    
    bark_mat_name = "Bark " + str(randint(1, 16))  # randomly among the 15 bark types
    params = tree_types.get(selected_tree).copy()
    if randint(1, 5) < 5:  # frequently randomize a new param set
        selected_tree = "Randomized"
        for param, ranges in param_ranges.items():
            if isinstance(ranges, tuple):
                _min, _max = ranges
                if (isinstance(_min, int) & isinstance(_max, int)):
                    rand_value = randint(_min, _max)
                else:
                    rand_value = round(uniform(_min, _max), 3)
            else:
                vals = []
                for _min, _max in ranges:
                    if (isinstance(_min, int) & isinstance(_max, int)):
                        elem_val = randint(_min, _max)
                    else:
                        elem_val = round(uniform(_min, _max), 3)
                    vals.append(elem_val)
                rand_value = vals
            params[param] = rand_value
    else:
        params['leaf_scale'] = (params.get('leaf_scale', 0.17) * 400) / 100
        params['leaf_scale_x'] = params.get('leaf_scale_x', 1) + 0.4
        leaf_bend = params.get('leaf_bend', 0.6)
        params['leaf_bend'] = uniform(leaf_bend - 0.2, leaf_bend + 0.2)

    # These param alterations attempt to convert the settings to make 'branches'
    #   effectivly bumping all paramaters down a level so the base is branch-like
    #   not trunklike
    skip_list = ['down_angle', 'down_angle_v', 'branches',
                 'tropism', 'curve_res']
    for param in params.keys():
        val = params[param]
        if isinstance(val, list) & (param not in skip_list):
            val.append(val[-1]) # repeat last values
            params[param] = val[1:] # reassign it with everything bumped down one

    # some params get manual intervention
    down_angle = params.get('down_angle', [-0, 70, 60, 45])
    down_angle[1] = down_angle[2]
    down_angle[2] = down_angle[3]
    params['down_angle'] = down_angle
    
    down_angle_v = params.get('down_angle_v', [-0, 10, 20, 30])
    down_angle_v[1] = down_angle_v[2]
    down_angle_v[2] = down_angle_v[3]
    params['down_angle_v'] = down_angle_v

    params['flare'] = 0
    levels = params['levels']
    if levels >2:
        params['levels'] -= 1

    branch_nums = params['branches']
    if randint(1, 7) == 1:
        # rarely, make a really busy specimen
        branch_nums[0] = randint(2, 3)
    else:
        branch_nums[0] = 1
    params['branches'] = branch_nums

    # handle max dichotomous branches (splitting events) at each segment
    if not params.get("seg_splits", False):
        seg_1 = choice([randint(0,2), round(uniform(0, 2.0), 1)])
        seg_2 = choice([randint(0,2), round(uniform(0, 1.5), 1)])
        seg_3 = choice([0, randint(0,2), round(uniform(0, 2.0), 1)])
        seg_4 =choice([0, randint(0,2), round(uniform(0, 2.0), 1)])
        params['seg_splits'] = [seg_1, seg_2, seg_3, seg_4]

    params['blossom_rate'] = -1  # be sure to turn off blossoms
    params.pop("leaves", False)  # make sure unexpected keyword 'leaves' is omitted

    # enforce a part count ceiling
    params = enforce_complexity_ceil(params)
    
    # guard against endless "stem" creation while loops
    params['prune_ratio'] = 0
    params['prune_width'] = 0.5
    params['prune_width_peak'] = 0.5
    params['prune_power_low'] = 0.5
    params['prune_power_high'] = 0.5

    # always use the rectangle option. Leaf shape is determined by the material name    
    params['leaf_shape'] = 9
    
    leaf_mat_names = ["ovate_mat_1", "ovate_mat_2", "ovate_mat_3", 
                      "cordate_mat_1",
                      "palmate_mat_1",
                      "linear_mat_1",
                      "lobed_mat_1", "lobed_mat_2",
                      "compound_mat_1", "compound_mat_2", "compound_mat_3",
                      "reniform_mat_1",
                      "serrate_mat_1",
                      ]
    leaf_mat_name = choice(leaf_mat_names)  # pick a random leaf material from the above list.
    sp_params = [selected_tree, bark_mat_name, leaf_mat_name, params]
        
    return sp_params

def set_ior(material, min_ior=1.2, max_ior=1.7):
    ## Function to set IOR of a material +/- 0.25 of default 1.45 value.
    ## Blender 4.0 changed BSDF_PRINCIPLED to include an IOR value.
    ## Some older Imported materials have this set to 0.00, far from the default 1.45
    ## This function ensures the values are reasonable.

    if material.node_tree:
        for node in material.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                ior_value = uniform(min_ior, max_ior)
                node.inputs['IOR'].default_value = ior_value
    return material  # return altered material

def remix_params(params):
    """
        Given a set of tree-gen params, makes slight modifications while keeping the 
        general spirit of the structure and style. Useful when multiple cuttings 
        should exist on one paper.
    """
    
    branches = params['branches']
    if randint(1, 3) == 1:  # about 1/3rd of the time, change the base branch number
        branches[0] = choice([x for x in range(1, 5) if x != branches])
    for i in range(1,4):
        element_val = branches[i]
        wiggle_range = element_val // uniform(0.8, 2.5)
        new_val = max([0, element_val + wiggle_range])
        branches[i] = int(round(new_val,0))
    params['branches'] = branches
    
    if randint(1,3) < 3:  # frequently reduce the leaf_blos_num (it is intensive)
        leaf_blos_num = params['leaf_blos_num']
        new_val = max([2, leaf_blos_num // uniform(0.8, 2)])
        params['leaf_blos_num'] = new_val 
    
    if randint(1, 4) < 4:  # frequently modify the leaf scale a little to simulate sampling from different portions
        leaf_scale = params['leaf_scale']
        new_val = max(0.1, leaf_scale + uniform(-0.3, 0.3))
        params['leaf_scale'] = new_val
        leaf_scale_x = params['leaf_scale_x']
        new_val = max(0.1, leaf_scale_x + uniform(-0.125, 0.125))
        params['leaf_scale_x'] = new_val

    if randint(1, 3) < 3:  # frequently adjust the max height of the 'tree'
        g_scale = params['g_scale']
        #new_val = int(g_scale * uniform(-3, 3))
        new_val = g_scale + randint(-3, 3)
        params['g_scale'] = new_val
    
    params = enforce_complexity_ceil(params)

    return params

def make_specimen(addon_info, sp_params=False):
    """
    sp_params used for specimen type continuity
    """
    tg, _, _ = addon_info  # unpack addon_info
    rand_seed()  # generate a new random seed
    specimen = False

    if not sp_params:
        sp_params = get_sp_params(addon_info)
    selected_tree, bark_mat_name, leaf_mat_name, params = sp_params  # unpack specimen info

    # clean up after self
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = None

    #### Tree construction parameters
    try:
        print(selected_tree)
        print(params)
        tree = tg.parametric.gen.construct(params, generate_leaves=True)
        trunk = [x for x in tree.children if x.name.startswith("Trunk")][0]
        trunk.data.use_fill_caps = True
        bpy.context.view_layer.objects.active = tree
        tg.utilities.convert_to_mesh(bpy.context)  # tree-gen utility func to convert branch's curve
    # Resorting to catching "Exception" due to rare exception caused in:
    #  tree-gen/parametric/gen.py", line 57, Exception: Offset out of range: 1.0001 not between 0 and 1
    except (TypeError, ZeroDivisionError, ValueError, Exception) as e:
        print()
        print("EXCEPTION!:", e)
        print(params)
        print()
        tree = bpy.data.objects['Tree']
        for child in tree.children:
            bpy.data.objects.remove(child, do_unlink=True)
        bpy.data.objects.remove(tree, do_unlink=True)
        raise ValueError

    leaves = []
    branches = []
    curves = []
    for child in tree.children:
        name = child.name
        if name.startswith("Leaves"):
            leaves.append(child)
        if name.startswith("Branches"):
            branches.append(child)
        if name.startswith("Trunk"):
            trunk = child

    branches.append(trunk) # trunk should also be considered a 'branch'
    
    # verify that an object was actually produced
    if not trunk.data.vertices.values():
        # if not remove it, it's children and restart the loop
        for child in tree.children:
            bpy.data.objects.remove(child, do_unlink=True)
        bpy.data.objects.remove(tree, do_unlink=True)
        raise ValueError

    if len(leaves) > 0:
        leaves = leaves[0]  # leaves is one object
    else:
        leaves = False

    # this block "splinters" the base of the trunk
    # however, the boolean caused scaling and performance issues
    #if randint(1, 5)>1:
        #estimate the trunk's base width and randomly destroy some of the base
        #b = params.get('branches', [1, 50, 30, 10])[0]
        #g = params.get('g_scale', 13) + params.get('g_scale_v', 3)
        #l = params.get('length', [1, 0.3, 0.6, 0])[0]
        #r = params.get('ratio' , 0.015)
        #m = params.get('radius_mod', [1, 1, 1, 1])[0]
        #radius = sum([(g * l * r * m) for x in range(b)])
        #approx_base_width = radius +  radius + 0.1
        #splinter_trunk(trunk, approx_base_width)
    
    # Join up the trunk and branch objects for efficiency when assigning materials
    bpy.ops.object.select_all(action='DESELECT')
    first_branch = branches[0]
    bpy.context.view_layer.objects.active = first_branch
    for branch in branches:
        branch.select_set(True)
    bpy.ops.object.join()
    branch = first_branch

    # assign element materials
    assign_materials(branch, leaves, bark_mat_name, leaf_mat_name)
    bpy.ops.object.select_all(action='DESELECT')

    # Join remaining elements togeather for unified positioning
    first_child = tree.children[0]
    bpy.context.view_layer.objects.active = first_child
    tree_children = tree.children

    if len(tree_children) > 0:  # Guard to make sure the entire plant-like object is a single mutable entity.
        for element in tree.children:
            element.select_set(True)
        bpy.ops.object.join()

    first_child.name = "specimen"
    first_child.parent = None
    bpy.data.objects.remove(tree, do_unlink=True)
    specimen = first_child
    if specimen.type != "MESH":
        raise ValueError

    return specimen, sp_params  # Pass on the specimen and generation parameters for metadata recording.

def assign_materials(branch, leaves, bark_mat_name, leaf_mat_name):
    """
    expects a joined branch object, leaf objects and associated material names
    """
    bpy.ops.object.select_all(action='DESELECT')

    # select bark material based on the name
    bark_mat = bpy.data.materials.get(bark_mat_name)
    # once the bark material is selected, set the IOR using a uniform range (low, high)
    bark_mat = set_ior(bark_mat, 1.15, 1.55)

    # select leaf material based on the name
    leaf_mat = bpy.data.materials.get(leaf_mat_name)
    # once the leaf material is selected, set the IOR using a uniform range (low, high)
    leaf_mat = set_ior(leaf_mat, 1.25, 1.65)

    # assign materials to combined branch objects
    bpy.context.view_layer.objects.active = branch
    branch.data.materials.append(bark_mat)
    branch.data.uv_layers["UVMap"].name = "combinedUV"
    displacement_texture = bpy.data.textures.new("wood_texture", "VORONOI")
    displacement_texture.noise_scale = uniform(0.001, 0.01)
    displacement_texture.contrast = uniform(0.005, 0.035)
    displacement_mod = branch.modifiers.new("Displace", "DISPLACE")
    displacement_mod.texture = displacement_texture
    bpy.ops.object.modifier_apply(modifier='Displace')

    if leaves:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = leaves
        leaves.select_set(True)
        leaves.data.materials.append(leaf_mat)
        leaves.data.uv_layers["leavesUV"].name = "combinedUV"

    bpy.ops.object.select_all(action='DESELECT')

def splinter_trunk(trunk, approx_base_width):
    """
    This function is no longer called. Keeping it here in case I return to it later
    Caused issues where: boolean messed up topography such that subsequent scaling
    destroyed the entire trunk object. Also, caused marked slowdown at runtime. 
    """
    
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = None
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=32,
                                    y_subdivisions=32,
                                    size=approx_base_width,
                                    align='WORLD',
                                    location=(0, 0, 0),
                                    scale=(1, 1, 1))
    cut_plane = bpy.context.object
    cut_plane.name = 'cut_plane'
    splinter_plane_mod = cut_plane.modifiers.new('splinter', 'NODES')
    sp_node_group = bpy.data.node_groups['splinter_plane']
    splinter_plane_mod.node_group = sp_node_group
    cut_plane.rotation_euler[0] = 1.5708
    cut_plane.location[2] += approx_base_width/2

    bpy.ops.object.modifier_apply(modifier='splinter')

    bpy.context.view_layer.objects.active = trunk    
    bool_mod = trunk.modifiers.new('comb_splinter', 'BOOLEAN')
    bool_mod.solver = 'FAST'
    bool_mod.double_threshold = 1e-8
    bool_mod.use_hole_tolerant = True
    bool_mod.object = cut_plane
    bpy.ops.object.modifier_apply(modifier='comb_splinter')

    bpy.data.objects.remove(cut_plane, do_unlink=True)    

def plant_press(specimen):
    """
    unused function lattice transform method. Kept here in case I attempt to return to it later.
    The idea is to use a lattice to simulate the effects of a plant press as oppose to 
    linear scaling.
    """
    
    # be sure the dimensions are correct    
    bpy.ops.object.add(type='LATTICE', enter_editmode=False, align='WORLD')
    lattice_ob = bpy.context.view_layer.objects.active
    lattice = lattice_ob.data
    lattice.points_u = 5
    lattice.points_v = 5
    lattice.points_w = 5
    lattice.interpolation_type_u = 'KEY_LINEAR'
    lattice.interpolation_type_v = 'KEY_BSPLINE'
    lattice.interpolation_type_w = 'KEY_LINEAR'
    lattice_ob.rotation_euler[2] = 1.5708 # rotate lattice 90deg due to vertex indexing order
    lattice_ob.dimensions = specimen.dimensions
    # lattice has origin at center, tree at base, fix this for z location
    lattice_ob.location = specimen.location
    lattice_ob.location.z =+ specimen.dimensions.z/2
    lattice_ob.scale = lattice_ob.scale *1.2  # be sure it is contained
    lattice_ob.parent = specimen

    # establish a target offset from y center
    offset = (specimen.dimensions.y * uniform(0.1, 0.25)) / 4
    
    lat_mod = specimen.modifiers.new("Lattice", "LATTICE")  # make a lattice modifier for the specimen
    lat_mod.object = lattice_ob  # assign the lattice object to guide the modifier
    
    # gather all the lat points
    lat_points = [x for x in lattice_ob.data.points]                
    vert_containers = [ [], [], [], [], []]
    for x_start_step in [0,1,3,4]:
        for y_start_step in range(x_start_step, 25, 5):  # this should select 2 entire series of y axis elements
            for idx in range(y_start_step, 125, 25):
                pt = lat_points[idx]
                vert_containers[x_start_step].append(pt)

    bpy.context.object.rotation_euler[2] = -1.5708
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # vert_containers[2] is empty (center line WRT y axis)
    for mult, idx in enumerate([1,0]):    
        loc_offset = offset# * (mult+1)
        for pt in vert_containers[idx]:
            pt.co_deform = Vector((pt.co.x, pt.co.y  - loc_offset, pt.co.z))

    for mult, idx in enumerate([3,4]):
        loc_offset = offset# * (mult+1)
        for pt in vert_containers[idx]:
            pt.co_deform = Vector((pt.co.x, pt.co.y + loc_offset, pt.co.z))

    bpy.context.view_layer.objects.active = specimen

def clean_leaf_clipping(leaf_mat_name):
    """
    Checks if any leaves are clipping through the herb_paper.
    This method is distinct from position_leaves method, in that it is called after
    the specimen is positioned on the herb_paper. This is akin to final adjustments
    made while affixing the specimen to the herb_paper.
    """
    # deselect everything
    bpy.ops.object.select_all(action='DESELECT')

    # identify the herb_paper for reference
    mat = bpy.data.objects['black_mat']
    mat_z = mat.location.z + 0.000151  # Slight offset to avoid Z fighting.

    bpy.ops.object.select_pattern(pattern=f'{leaf_mat_name}*')
    leaf_objects = bpy.context.selected_objects

    for leaves in leaf_objects:
        # split the leaf mesh up into individual, disconnected parts (leaves)
        bpy.context.view_layer.objects.active = leaves
        leaves.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.separate(type='LOOSE')
        bpy.ops.object.mode_set(mode='OBJECT')

    kept_leaves = []
    for leaf in bpy.context.selected_objects:  # should result in each individual leaf object being selected
        # nested comprehension, inner comp retrieves the z coord of the leaf's (l) vertices
        #    outer comp filters the inner's results by those above the herb_paper_z
        #    ref: https://blender.stackexchange.com/a/73702
        verts_above_paper = [z for z in [(leaf.matrix_world @ v.co).z for v in leaf.data.vertices] if z > mat_z]
        len_verts_above_paper = len(verts_above_paper)

        if len_verts_above_paper < 2:
            # if none, or only 1 are above the paper, consider it a lost cause and remove it.
            bpy.data.objects.remove(leaf, do_unlink=True)
        elif len_verts_above_paper < 4:
            # if 2, 3 verts are above the paper (but not all) orient with shrink wrapping
            # snap the leaves to the black mat
            bpy.context.view_layer.objects.active = leaf
            shrink_mod = leaf.modifiers.new('shrinkwrap', 'SHRINKWRAP')
            shrink_mod.target = mat
            shrink_mod.wrap_mode = 'OUTSIDE_SURFACE'
            # Make sure it rests above, paper.
            # Random offset helps avoid leaves overlapping with rotation and position
            shrink_mod.offset = uniform(0.001, 0.02)
            bpy.ops.object.modifier_apply(modifier='shrinkwrap')
            kept_leaves.append(leaf)
        else:
            # otherwise all vertices are over the paper.. don't mess with it.
            kept_leaves.append(leaf)

    # now if necessary, re-merge the leaves into a single mesh object
    if len(kept_leaves) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        parent_leaf = kept_leaves[0]
        bpy.context.view_layer.objects.active = parent_leaf
        for leaf in kept_leaves:
            leaf.select_set(True)
        bpy.ops.object.join()

def position_specimen(specimen):

    # Squash the object along x axis
    #plant_press(specimen)  # no longer called in favor of linear scaling    
    bpy.context.view_layer.objects.active = specimen
    # scale along y axis to "press" the specimen.
    bpy.context.object.scale[1] = uniform(0.07, 0.15)

    bpy.context.view_layer.objects.active = specimen
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')  # change element origins to their geometric center

    #  Prepare for resizing the object
    largest_dim = max(specimen.dimensions)
    # largest dim ~0.4 for paper
    scale_factor = uniform(0.4, 0.95) / largest_dim
    # resize specimen
    bpy.ops.transform.resize(value=(scale_factor,scale_factor,scale_factor))

    # rotate the specimen
    x = 1.5708  #i.e., radians(90)
    y = 0
    z = radians(choice([uniform(-90, -60), uniform(60, 90)]))
    specimen.rotation_euler[0:3] = [x,y,z]

    # Handle the object's starting location
    # id the herb_paper's location
    paper = bpy.data.objects['herb_paper']
    x_reach = paper.dimensions.x / 2
    y_reach = paper.dimensions.y / 2
    min_x = paper.location.x - x_reach
    max_x = paper.location.x + x_reach
    min_y = paper.location.y - y_reach
    max_y = paper.location.y + y_reach
    # get the average between x, y dim.. this will cause minimal occasional overlap
    #    which sometimes happens in specimens
    specimen_reach = (specimen.dimensions.x + specimen.dimensions.y) / 2.2
    min_x += specimen_reach
    max_x -= specimen_reach
    min_y += specimen_reach
    max_y -= specimen_reach

    # if multiple specimens are in the scene, sort of try to keep them apart.
    existing_specimen_list = [x for x in bpy.data.objects if x.name.startswith("specimen")]
    if existing_specimen_list:
        distances = []
        for i in range(10):
            x = uniform(min_x, max_x)
            y = uniform(min_y, max_y)
            z = uniform(0.008, 0.015)
            proposed_pos = Vector([x,y,z])
            sum_distances = sum([(proposed_pos - x.location).length for x in existing_specimen_list])
            distances.append((sum_distances, proposed_pos))
        distances.sort(reverse=True)
        proposed_pos = distances[0][-1]
    else:
        x = uniform(min_x, max_x)
        y = uniform(min_y, max_y)
        z = uniform(0.01, 0.015)
        proposed_pos = Vector([x,y,z])

    specimen.location = proposed_pos

    # clean up after self
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = None

def split_by_material(bark_mat_name, leaf_mat_name):
    """
    splits the specimen objects up by their associated material names
    intended to be called after unified positioning (via position_specimen)
    """
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_pattern(pattern='specimen*')
    for specimen in bpy.context.selected_objects:
        # split each specimen up into its material elements (leaves, branches)
        bpy.context.view_layer.objects.active = specimen
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.separate(type='MATERIAL')
        bpy.ops.object.mode_set(mode='OBJECT')

    # by this point all material-named objects should be selected
    specimen_elements = bpy.context.selected_objects
    for specimen_element in specimen_elements:
        specimen_element.hide_render = False  # better safe than confused I suppose
        active_mat_name = specimen_element.active_material.name
        specimen_element.name = active_mat_name
        
        if active_mat_name.startswith(bark_mat_name):
            # set the object's pass Index based on the class_ids dict
            specimen_element.pass_index = class_ids.get("bark", 0)
        elif active_mat_name.startswith(leaf_mat_name):
            # set the object's pass Index based on the class_ids dict
            specimen_element.pass_index = class_ids.get("leaf", 0)


def create_specimens(addon_info, guarantee_single=False):
    """
    function to call the specimen creation functions in order
    handles generating, positioning, material splitting, and cleaning leaf clipping

    guarantee_single is useful when debugging as it only ever produces one specimen.
    """

    # reset the scene, and set back to normal, then nudge it a little
    working = True
    while working:
        try:
            reset_scene()
            reset_set()
            sample_gen_params = randomize_set()  # retrieve prop info such as crc_type, scale, etc.

            # build the first specimen
            specimen, sp_params = make_specimen(addon_info)
            if not sp_params:
                # if specimen generation produced impossible tree-gen params
                # the result would be an endless loop, so instead:
                # break iteration cycle reset the scene and try again
                raise ValueError

            selected_tree, bark_mat_name, leaf_mat_name, params = sp_params  # unpack specimen info
            sample_gen_params['selected_tree'] = selected_tree
            sample_gen_params['bark_mat_name'] = bark_mat_name
            sample_gen_params['leaf_mat_name'] = leaf_mat_name
            position_specimen(specimen)  # position the specimen

            if not guarantee_single:    
                # maybe add additional specimens to the scene
                roll_over = randint(1, 4)
                roll_count = 0  # it gets less and less likely to add more specimens on top.
                while  roll_over < 2:
                    roll_count += 1
                    # last element in sp_params is the tree-gen params, remix them
                    sp_params[-1] = remix_params(sp_params[-1])
                    specimen, sp_params = make_specimen(addon_info, sp_params)
                    if not specimen:
                        # again, if impossible params exist then break the iteration
                        # cycle,  reset the scene and try again
                        raise ValueError
                    position_specimen(specimen)
                    roll_over = randint(1, 2+roll_count)
            working = False  # seems like everything finished fine
        except ValueError:
            working = True    

    # after the specimens are created...
    split_by_material(bark_mat_name, leaf_mat_name)  # split them up by material
    clean_leaf_clipping(leaf_mat_name)  # and clean up clipping issue

    bpy.context.view_layer.update()  # guard against rare texture errors

    return sample_gen_params, sp_params

def dump_json_dict(input_dict, fn):
    with open(fn, 'w') as out_file:
        json.dump(input_dict, out_file, sort_keys = True, indent = 4, ensure_ascii = False)

def clean_up_scene(elements):
    """ Given a list of element prefixes, attempts to purge them from the scene.
        Distinct from reset_scene() in that it is targeted, and does not attempt
        to clear orphans or partially created (in process) objects.
    """
    bpy.ops.object.select_all(action='DESELECT')
    for element in elements:
        bpy.ops.object.select_pattern(pattern=f'{element}*')
    bpy.data.batch_remove(bpy.context.selected_objects)

############################
# Actual script processes: #
############################

# open the backdrop scene, from a light_box blend file
bpy.ops.wm.open_mainfile(filepath="./light_box.blend")

# inital set up for addon_info, one time at start-up
addon_info = setup_addon()

debug = False
if debug:
    sample_gen_params, sp_params = create_specimens(addon_info, guarantee_single=True)

else:
    # get rough estimate of runtime
    start = time.time()
    qty = 10  # the number of additional scenes to produce

    # get a handle on each output's file name. Do this each iteration ...
    # it changes each iteration, so re run to avoid seg faults
    scene = bpy.context.scene
    output_node = scene.node_tree.nodes['File Output']
    file_slots = [x for x in output_node.file_slots]
    #  This will need updated in the blend file to run in a new environment.
    dst_folder = output_node.base_path  # set up a default dst folder
    start_num = (len(glob(f"{dst_folder}/*.jpg")))  # count how many scenes already exist
    for i in range(start_num, start_num+qty):
        # create the specimen(s) for the scene
        sample_params, sp_params = create_specimens(addon_info)
        selected_tree, bark_mat_name, leaf_mat_name, params = sp_params  # unpack specimen info
        # setup the output names
        file_slots[0].path = f"{i}_src_"  # The entire scene "image" (i.e., this looks like a plant specimen image)
        file_slots[1].path = f"{i}_msk_"  # The annotation map for this scene (i.e., the segmentation training data)

        # write out the json details
        sample_param_fn = f"{dst_folder}/{i}_details.json"
        dump_json_dict(sample_params, sample_param_fn)

        # destination path is determined in the scene ".blend" file.
        bpy.ops.render.render()  # render out the scene
        # clean up the scene's material elements
        clean_up_scene([bark_mat_name, leaf_mat_name])

    runtime = time.time() - start
    print("Runtime (seconds): ", runtime)
    print("seconds per specimen: ", runtime / qty)
