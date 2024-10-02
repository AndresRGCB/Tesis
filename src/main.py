bl_info = {
    "name" : "GraspQuality map",
    "author" : "Andres Robles Gil",
    "version" : (1,0),
    "blender" : (4, 0, 0),
    "location" : "View3d > Tool",
    "category" : "Add mesh"
}

import bpy
import os
import sys
import subprocess
import json
import random  # Ensure random is imported
import pandas as pd  # Explicit import for pandas
from pathlib import Path  # Ensure Path is imported
from datetime import datetime  # Explicit import for datetime
import math
import bmesh
import numpy as np
import mathutils
import glob
import re
import shutil
import matplotlib.image as mpimg
import cv2
import OpenEXR
import Imath
from PIL import Image
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
import imageio
from skimage.morphology import disk
from scipy.ndimage import binary_dilation, convolve
import csv




# ----------------------------------------------- PANELS ----------------------------------------------------------------

#Main Panel
class UI_PT_main(bpy.types.Panel):
    bl_label = "GQ dataset generator"
    bl_idname = "UI_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GQ dataset generator"
    
    def draw(self, context):
        layout = self.layout
        
        layout.operator("object.generate_dataset", text="Generate Dataset")
        
#Import models filepath        
class MODEL_PT_Panel(bpy.types.Panel):
    bl_label = "Model selection"
    bl_idname = "MODEL_PT_Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GQ dataset generator"
    bl_parent_id = "UI_PT_main"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        
        # Display the current object_quality directory path
        layout.prop(context.scene, "object_quality_directory", text="Objects Directory")
        
        # Display the current bin_quality directory path
        layout.prop(context.scene, "bin_quality_directory", text="Bin Directory")
        
        # Display the current bin_quality directory path
        layout.prop(context.scene, "materials_directory", text="Material JSON")


#Scene paremeter settings
class PARAMETERS_PT_Panel(bpy.types.Panel):
    bl_label = "Scene Parameters"
    bl_idname = "PARAMETERS_PT_Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GQ scene generator"
    bl_parent_id = "UI_PT_main"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        
        
        #Bin Settings
        box = layout.box()
        box.label(text="Bin", icon='OUTLINER_COLLECTION')
        box.label(text="Bin Scaling")
        row = box.row()
        row.prop(context.scene, "min_bin_scaling", text="Minimum Scaling")
        row.prop(context.scene, "max_bin_scaling", text="Maximum Scaling")
        box.prop(context.scene, "bin_rotation", text="Bin Rotation")
        
        
        
        #Objects in Scene
        box1 = layout.box()
        box1.label(text="Objects", icon='MESH_CUBE')
        
        box1.label(text="Objects per Scene")
        row = box1.row()
        row.prop(context.scene, "min_objects", text="Minimum Objects")
        row.prop(context.scene, "max_objects", text="Maximum Objects")
        
        box1.prop(context.scene, "scaling", text="Enable Scaling")
        
        if context.scene.scaling:
            box1.label(text="Object Scaling")
            row1 = box1.row()
            row1.prop(context.scene, "min_scaling", text="Minimum Scaling")
            row1.prop(context.scene, "max_scaling", text="Maximum Scaling")
        
        
        #Lighting
        box2 = layout.box()
        box2.label(text="Lighting", icon='OUTLINER_OB_LIGHT')
        box2.prop(context.scene, "max_lights", text="Maximum Light Sources")
        box2.prop(context.scene, "max_light_strength", text="Maximum Light Strength")
        box2.prop(context.scene, "color_light", text="Random Colored Lights")
        
        
        #Camera
        box3 = layout.box()
        box3.label(text="Camera", icon='CAMERA_DATA')
        
        box3.prop(context.scene, "max_camera_angle", text="Maximum Camera Angle")
        
        box3.label(text="Camera Height")
        row3 = box3.row()
        row3.prop(context.scene, "min_camera", text="Minimum Height")
        row3.prop(context.scene, "max_camera", text="Maximum Height")
        
        box3.label(text="Camera Intrinsics")
        box3.prop(context.scene, "res_x", text="Resolution x")
        box3.prop(context.scene, "res_y", text="Resolution y")
        box3.prop(context.scene, "focal_length", text="Focal Lenght")
        box3.prop(context.scene, "sensor_width", text="Sensor width")
        box3.prop(context.scene, "sensor_height", text="Sensor Height")
        
        box3.label(text="Stereo Image")
        box3.prop(context.scene, "stereo", text="Stereo Image")
        
        
        
#Select Output files      
class OUTPUT_PT_Panel(bpy.types.Panel):
    bl_label = "Output"
    bl_idname = "OUTPUT_PT_Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GQ dataset generator"
    bl_parent_id = "UI_PT_main"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        
        box = layout.box()
        box.label(text="Rendering", icon='RENDERLAYERS')
        row = box.row()
        col1 = row.column(align=True)
        col2 = row.column(align=True)

        col1.prop(context.scene, "rgb", text="RGB")
        col1.prop(context.scene, "depth", text="Depth Map")
        col1.prop(context.scene, "normals", text="Normal Vectors")

        col2.prop(context.scene, "segmentation", text="Segmentation")
        
        # Directly check the conditions in the draw method
        if context.scene.rgb and context.scene.depth and context.scene.normals and context.scene.segmentation:
            col2.prop(context.scene, "gq", text="Grasp Quality Map")


        box2 = layout.box()
        box2.label(text="Extras", icon='EXPERIMENTAL')
        if context.scene.gq:
            box2.prop(context.scene, "gq_elements", text="GQ Elements")
        box2.prop(context.scene, "positions", text="Positions")
        
        box3 = layout.box()
        box3.label(text="Output", icon='DOWNARROW_HLT')
        box3.prop(context.scene, "output_directory", text="Output Directory")
        box3.prop(context.scene, "dataset_size", text="Dataset size")
        box3.prop(context.scene, "starting_scene", text="Starting Scene Number")
        
        


#Stereo Settings
class STEREO_PT_Panel(bpy.types.Panel):
    bl_label = "Stereo Image Settings"
    bl_idname = "STEREO_PT_Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GQ dataset generator"
    bl_parent_id = "UI_PT_main"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        return context.scene.stereo
    
    def draw(self, context):
        layout = self.layout
        layout.label(text="Stereo Settings")
        
        box = layout.box()
        box.prop(context.scene, "camera_baseline", text="Camera Baseline(cm)")
        
        box.prop(context.scene, "matrix", text="IR Matrix")
        
        if context.scene.matrix:
            box.prop(context.scene, "ir_side", text="Matrix size")
            box.prop(context.scene, "ir_separation", text="Dot Separation (Degrees)")
            

        
#Grasp Quality settings
class GQ_PT_Panel(bpy.types.Panel):
    bl_label = "Grasp Quality Settings"
    bl_idname = "GQ_PT_Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GQ dataset generator"
    bl_parent_id = "UI_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        # Check if all required conditions for the Grasp Quality checkbox to appear are met
        prerequisites_met = (context.scene.rgb and context.scene.depth and
                             context.scene.normals and context.scene.segmentation)
        # The panel appears only if Grasp Quality is selected and all prerequisites are met
        return context.scene.gq and prerequisites_met

    def draw(self, context):
        layout = self.layout
        
        box1 = layout.box()
        box1.prop(context.scene, "dist_center", text="Distance to Center of Mass", icon='SNAP_FACE_CENTER')
        if context.scene.dist_center:
            box1.prop(context.scene, "weight_distance", text="Weight")

        box2 = layout.box()
        box2.prop(context.scene, "gq_height", text="Height", icon='EMPTY_SINGLE_ARROW')
        if context.scene.gq_height:
            box2.prop(context.scene, "weight_height", text="Weight")
            
        box3 = layout.box()
        box3.prop(context.scene, "gq_flatness", text="Flatness", icon='RNDCURVE')
        if context.scene.gq_flatness:
            box3.prop(context.scene, "weight_flatness", text="Weight")
            box3.prop(context.scene, "flat_log", text="Exponent")
            
        box4 = layout.box()
        box4.prop(context.scene, "gq_smoothness", text="Smoothness", icon='SMOOTHCURVE')
        if context.scene.gq_smoothness:
            box4.prop(context.scene, "weight_smoothness", text="Weight")
            box4.prop(context.scene, "smooth_log", text="Exponent")
            
        box5 = layout.box()
        box5.prop(context.scene, "gq_impacts", text="Ray Tracing", icon='LIGHT_AREA')
        if context.scene.gq_impacts:
            row10 = box5.row()
            row10.prop(context.scene, "impacts_mask", text="Mask")
            row10.prop(context.scene, "impacts_weight_bool", text="Weight")
            if context.scene.impacts_weight_bool:
                box5.prop(context.scene, "weight_impacts", text="Weight")
            
        box6 = layout.box()
        box6.prop(context.scene, "gq_edge", text="Edge Detection", icon='SNAP_VOLUME')
        if context.scene.gq_edge:
            row11 = box6.row()
            row11.prop(context.scene, "edge_mask", text="Mask")
            row11.prop(context.scene, "edge_weight_bool", text="Weight")
            if context.scene.edge_weight_bool:  # Corrected from impacts_edge_weight_bool to edge_weight_bool
                box6.prop(context.scene, "weight_edge", text="Weight")
            box6.prop(context.scene, "cup_diameter", text="Suction Cup Diameter (mm)")
            box6.prop(context.scene, "edge_angle", text="Edge Angle (°)")
            
            
            
        
        
#Segmentation Mask settings
class SEGMENTATION_PT_Panel(bpy.types.Panel):
    bl_label = "Segmentation Settings"
    bl_idname = "SEGMENTATION_PT_Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GQ dataset generator"
    bl_parent_id = "UI_PT_main"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        return context.scene.segmentation
    
    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.prop(context.scene, "segmentation_mode", text="Segmentation Mode")
        
        if context.scene.segmentation_mode == 'INSTANCE':
            box.prop(context.scene, "bounding_boxes", text="Bounding Boxes")


# ------------------------------------ PROPERTIES ----------------------------------------------------------------
        
def update_impacts_mask(self, context):
    # When impacts_mask is selected, uncheck impacts_weight_bool
    if not(context.scene.impacts_mask):
        context.scene.impacts_weight_bool = True
    else:
        # If both properties are unselected, select impacts_mask
        context.scene.impacts_mask = False

def update_impacts_weight(self, context):
    # When impacts_weight_bool is selected, uncheck impacts_mask
    if not(context.scene.impacts_weight_bool):
        context.scene.impacts_mask = True
    else:
        # If both properties are unselected, select impacts_weight_bool
        context.scene.impacts_weight_bool = False
        
def update_edge_mask(self, context):
    if not(context.scene.edge_mask):
        context.scene.edge_weight_bool = True
    else:
        context.scene.edge_mask = False

def update_edge_weight(self, context):
    if not(context.scene.edge_weight_bool):
        context.scene.edge_mask = True
    else:
        context.scene.edge_weight_bool = False

        
def update_min_objects(self, context):
    if self.min_objects > self.max_objects:
        self.max_objects = self.min_objects

def update_max_objects(self, context):
    if self.max_objects < self.min_objects:
        self.min_objects = self.max_objects
        
def update_min_scaling(self, context):
    if self.min_scaling > self.max_scaling:
        self.max_scaling = self.min_scaling

def update_max_scaling(self, context):
    if self.max_scaling < self.min_scaling:
        self.min_scaling = self.max_scaling
        
def update_min_camera(self, context):
    if self.min_camera > self.max_camera:
        self.max_camera = self.min_camera

def update_max_camera(self, context):
    if self.max_camera < self.min_camera:
        self.min_camera = self.max_camera
        
def update_min_bin_scaling(self, context):
    if self.min_bin_scaling > self.max_bin_scaling:
        self.max_bin_scaling = self.min_bin_scaling

def update_max_bin_scaling(self, context):
    if self.max_bin_scaling < self.min_bin_scaling:
        self.min_bin_scaling = self.max_bin_scaling
        
def update_weight_distance(self, context):
    context.scene.weight_height = 1.0 - self.weight_distance

def update_weight_height(self, context):
    context.scene.weight_distance = 1.0 - self.weight_height


        
# ------------------------------------ OPERATORS ----------------------------------------------------------------
        

def segmentation_modes_callback(scene, context):
    return [
        ('INSTANCE', "Instance Segmentation", "Use instance segmentation"),
        ('SEMANTIC', "Semantic Segmentation", "Use semantic segmentation"),
    ]


class FolderFinder(bpy.types.Operator):
    bl_idname = "select.folder"
    bl_label = "Select Folder/File"

    # Properties for directory and file path selection
    directory: bpy.props.StringProperty(subtype="DIR_PATH")
    file: bpy.props.StringProperty(subtype="FILE_PATH")  # Added for file selection
    directory_key: bpy.props.StringProperty()

    def execute(self, context):
        # Handle the selection based on the directory_key
        if self.directory_key == "object_quality":
            context.scene.object_quality_directory = self.directory
        elif self.directory_key == "bin_quality":
            context.scene.bin_quality_directory = self.directory
        elif self.directory_key == "output_directory":
            context.scene.output_directory = self.directory
        elif self.directory_key == "materials_directory":  # Handle materials file
            context.Scene.materials_directory = self.file

        return {'FINISHED'}

    def invoke(self, context, event):
        # Check if it's a file or directory selection based on the directory_key
        if self.directory_key == "materials_directory":
            context.window_manager.fileselect_add(self)  # For file selection (e.g., JSON)
        else:
            context.window_manager.fileselect_add(self)  # For directory selection

        return {'RUNNING_MODAL'}

    
# Function to fill a DataFrame with file names and types from a given directory
def fill_dataframe_from_directory(directory_path):
    # Convert the directory path to a Path object
    dir_path = Path(directory_path)
    
    # Define the desired file extensions
    desired_extensions = {'.obj', '.ply', '.stl'}

    # Use rglob to recursively find all files
    files = dir_path.rglob('*.*')  # This will consider all file types initially

    # Filter files by desired extensions and gather data
    data = [{
        'name': file.name,
        'type': file.suffix.lower(),  # Get the file extension and convert it to lower case
        'full_path': file.resolve().as_posix()  # Get the full path in POSIX format
    } for file in files if file.is_file() and file.suffix.lower() in desired_extensions]

    # Create a DataFrame from the list
    df = pd.DataFrame(data, columns=['name', 'type', 'full_path'])
    
    return df

def create_or_update_material_from_json(material_info):
    material_name = material_info['name']
    
    # Check if the material exists in the Blender file
    if material_name in bpy.data.materials:
        material = bpy.data.materials[material_name]
    else:
        # Create a new material if not found
        material = bpy.data.materials.new(name=material_name)

    # Ensure the material uses nodes
    material.use_nodes = True
    nodes = material.node_tree.nodes

    # Find or create the Principled BSDF node
    bsdf = None
    for node in nodes:
        if node.type == 'BSDF_PRINCIPLED':
            bsdf = node
            break
    if not bsdf:
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')

    # Update Base Color
    if 'base_color' in material_info:
        bsdf.inputs['Base Color'].default_value = material_info['base_color'] + [1.0]  # Adding alpha channel

    # Update Roughness
    if 'roughness' in material_info:
        bsdf.inputs['Roughness'].default_value = material_info['roughness']

    # Update Metallic
    if 'metallic' in material_info:
        bsdf.inputs['Metallic'].default_value = material_info['metallic']

    # Optional additional properties from your JSON file
    # Add more properties if needed based on the JSON structure

    print(f"Material '{material_name}' has been created/updated.")




def generate_scene(objects, bins, context, scene):
    # Remove existing materials
    materials = bpy.data.materials
    for material in materials:
        bpy.data.materials.remove(material)

    # Load material data from the JSON file
    json_file_path = bpy.context.scene.materials_directory  # Replace with your JSON file path

    with open(json_file_path, 'r') as json_file:
        material_data = json.load(json_file)

    # Create or update materials based on JSON data
    for mat_info in material_data:
        create_or_update_material_from_json(mat_info)
    
    
    
    
    
    #Set simulation to 1 frame
    bpy.context.scene.frame_set(1)
    material_names=["1970_tiles", "Ceramic", "Crushed Velvet Eevee", "Polished Walnut", "WoodP", "wood", "wood boards"]
    
    #Delete current objects
    for obj in bpy.data.objects:
        obj.select_set(True)
    bpy.ops.object.delete()
    
    #Intialize object counter
    object_counter = 1

    
    #Introduce Bin
    random_bin = bins.sample(n=1).iloc[0]
    bin_name = random_bin['name']
    print(f'SE USO ESTE BIN {bin_name}')
    
    bin_path = context.scene.bin_quality_directory
    object_path = context.scene.object_quality_directory
    
    bpy.ops.import_mesh.stl(filepath = bin_path + str(bin_name), global_scale=0.1)
    imported_object = bpy.context.view_layer.objects.active
    imported_object.name = "Bin"
    bpy.ops.rigidbody.object_add()
    imported_object.rigid_body.type = 'ACTIVE'
    imported_object.rigid_body.type = 'PASSIVE'
    imported_object.rigid_body.collision_shape = 'MESH'
    imported_object.rigid_body.collision_margin = 0.1 
    imported_object.pass_index = object_counter


    def assign_random_material(imported_object, material_data):
        # Choose a random material from the list of material data (JSON)
        random_mat_info = random.choice(material_data)
        material_name = random_mat_info['name']
        
        # Retrieve the material from Blender's materials data
        material = bpy.data.materials.get(material_name)
        
        if material:
            # Check if there are any materials on the imported object
            if len(imported_object.data.materials) == 0:
                # If no materials, append the new material
                imported_object.data.materials.append(material)
            else:
                # If materials exist, replace the first material slot
                imported_object.data.materials[0] = material
            print(f"Assigned material '{material_name}' to object '{imported_object.name}'")
        else:
            print(f"Material '{material_name}' not found in Blender's data.")

    # Example Usage:
    # Assuming 'imported_object' is the object you want to assign a material to
    # and 'material_data' is the loaded data from the JSON file

    json_file_path = bpy.context.scene.materials_directory

    with open(json_file_path, 'r') as json_file:
        material_data = json.load(json_file)

    # Assuming 'imported_object' is your object
    imported_object = bpy.context.active_object  # Use the currently selected object, or replace with the target object

    # Assign a random material from JSON data to the imported object
    assign_random_material(imported_object, material_data)
                
    
    random_scale_factor = random.uniform(context.scene.min_bin_scaling, context.scene.max_bin_scaling)
    imported_object.scale = (random_scale_factor, random_scale_factor, random_scale_factor)
    bpy.context.view_layer.objects.active = imported_object
    if (context.scene.bin_rotation):
        imported_object.rotation_euler[2] = math.radians(random.uniform(0, 360))
    else:
        imported_object.rotation_euler[2] = 0
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)


    
    #Introduce objects
    items_scene = random.randint(context.scene.min_objects, context.scene.max_objects)
    random_objects = objects.sample(n=items_scene, replace=True)
    for index, row in random_objects.iterrows():
        object_name = row['name']
        _, extension = os.path.splitext(object_name)
        full_path = row['full_path']
        extension = extension[1:] 
        if extension == 'stl':
            bpy.ops.import_mesh.stl(filepath= full_path)
        elif extension == 'ply': 
            bpy.ops.wm.ply_import(filepath= full_path)
        elif extension == 'obj':
            bpy.ops.wm.obj_import(filepath= full_path)
        
        
        imported_object = bpy.context.view_layer.objects.active
        imported_object.name = object_name.split('.', 1)[0]+str(object_counter)
        object_counter += 1
        
        
        bm = bmesh.new()
        bm.from_mesh(imported_object.data)
        volume = bm.calc_volume(signed=True)
        bm.free() 
        
        if extension == 'ply':
            desiered_scale = 5000
        elif extension == 'obj':
            desiered_scale = 5000
        else:
            desiered_scale = 5000
            
        size_factor=(desiered_scale/volume)**(1/3)
        scale_factor = 1.0
        
        #Set scale
        if (context.scene.scaling):
            scale_factor = random.uniform(context.scene.min_scaling, context.scene.max_scaling)
            
        imported_object.scale = (scale_factor*size_factor, scale_factor*size_factor, scale_factor*size_factor)
            
        #Set material
        mat_names=["1970_tiles", "Ceramic", "Crushed Velvet Eevee", "Polished Walnut", "WoodP", "wood", "wood boards"]
        random_integer = random_integer = np.random.randint(0, 6)
        random_mat = material_names[random_integer]
        material = bpy.data.materials.get(random_mat)
        if material:
            # Check if there are any materials on the imported object
            if len(imported_object.data.materials) == 0:
                # If no materials, append the new material
                imported_object.data.materials.append(material)
            else:
                # If materials exist, replace the first material slot
                imported_object.data.materials[0] = material
    
    
        #Set position
        position_x = random.uniform(-20*random_scale_factor, 20*random_scale_factor)  
        position_y = random.uniform(-20*random_scale_factor, 20*random_scale_factor)  
        position_z = random.uniform(10, 1500)   
        imported_object.location = (position_x, position_y, position_z)
        
        #Set as rigid body
        bpy.ops.rigidbody.object_add()
        imported_object.rigid_body.type = 'ACTIVE'
        imported_object.rigid_body.mass = 1
        imported_object.rigid_body.linear_damping = 0.5  
        imported_object.rigid_body.angular_damping = 0.5  
        
        #set objects origin as center of mass
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME')
          
        
        #Set rotation
        rotation_x = random.uniform(0, 2 * math.pi) 
        rotation_y = random.uniform(0, 2 * math.pi)  
        rotation_z = random.uniform(0, 2 * math.pi) 
        imported_object.rotation_euler = (rotation_x, rotation_y, rotation_z)
        
        if context.scene.segmentation_mode == 'INSTANCE':
            imported_object.pass_index = object_counter  
        else: 
            imported_object.pass_index = index+2
        
    # Create plane 
    bpy.ops.mesh.primitive_plane_add(size=500, location=(0, 0, -0.1))
    plane = bpy.context.active_object
    random_mat = random.choice(material_names)
    material = bpy.data.materials.get(random_mat)
    if material:
        if len(plane.data.materials) == 0:
            plane.data.materials.append(material)
        else:
            plane.data.materials[0] = material
        
    #Drop objects
    bpy.context.scene.frame_end = 3500
    bpy.context.scene.rigidbody_world.point_cache.frame_start = 1
    bpy.context.scene.rigidbody_world.point_cache.frame_end = 3500
    for frame in range(1, 3501):  
        bpy.context.scene.frame_set(frame)  
        
        
        
    #Set lights
    lights_scene = random.randint(1, context.scene.max_lights)
    for i in range(lights_scene):
        point_light_data = bpy.data.lights.new(name="MyPointLight" + str(i), type='POINT')
        
        # Random light strength
        point_light_data.energy = random.randint(50*100000, int(context.scene.max_light_strength)*100000)
        
        if (context.scene.color_light):
            random_red = random.randint(0, 255) / 255.0
            random_green = random.randint(0, 255) / 255.0
            random_blue = random.randint(0, 255) / 255.0
            point_light_data.color = (random_red, random_green, random_blue)
        else:
            point_light_data.color = (1.0, 1.0, 1.0)
            
        
        # Create a new object with the light data
        point_light_object = bpy.data.objects.new(name="MyPointLightObj" + str(i), object_data=point_light_data)
        bpy.context.collection.objects.link(point_light_object)
        
        # Set random location for the light
        light_x = random.uniform(-200, 200)
        light_y = random.uniform(-200, 200)
        light_z = random.uniform(200, 1000)
        point_light_object.location = (light_x, light_y, light_z)
    
    #Set camera
    camera_data = bpy.data.cameras.new(name='Camera')
    camera_object = bpy.data.objects.new('MyCameraObject', camera_data)
    bpy.context.collection.objects.link(camera_object)
    camera_object.data.lens = bpy.context.scene.focal_length
    camera_object.data.sensor_width = bpy.context.scene.sensor_width
    camera_object.data.sensor_height = bpy.context.scene.sensor_height  
    camera_object.data.type = 'PERSP'
    
    if bpy.context.scene.stereo:
        camera_object.data.stereo.convergence_mode = 'PARALLEL'
        camera_object.data.stereo.interocular_distance = bpy.context.scene.camera_baseline  # Adjust as needed (average human eye distance)
    
    def create_laser_grid(name, location, grid_size, angle_step, rotation):
        half_grid = grid_size // 2
        for i in range(grid_size):
            for j in range(grid_size):
                # Create the spotlight
                bpy.ops.object.light_add(type='SPOT', radius=1, location=location)
                light = bpy.context.object
                light.name = f"{name}_{i}_{j}"
                
                # Calculate the x and y angle offsets
                x_angle_offset = (i - half_grid) * angle_step
                y_angle_offset = (j - half_grid) * angle_step

                # Set the rotation so that the lasers spread in a grid pattern
                light.rotation_euler = (rotation[0] + y_angle_offset, rotation[1] + x_angle_offset, rotation[2])
                
                # Configure the spotlight to simulate a laser
                light.data.energy = 1000000  # Adjust energy to avoid overwhelming brightness
                light.data.spot_size = math.radians(0.01)  # Keep the beam very narrow
                light.data.spot_blend = 0  # No blending for a sharp beam
                light.data.color = (1, 0, 0)  # Red color
                light.data.color = (1, 0, 0)  # Red color
                light.data.spot_blend = 10  # No blending for a sharp beam

                # Set the custom distance to elongate the beam
                light.data.use_custom_distance = True
                light.data.cutoff_distance = 500  # Adjust distance to elongate the spotlight beam
    
    
    theta_angle = 90-(random.randint(1, context.scene.max_camera_angle+1))
    camera_distance = random.uniform(context.scene.min_camera*100, context.scene.max_camera*100) 
    
    z_camera=math.sin(math.radians(theta_angle))*camera_distance
    beta_angle = 0
    x_camera=(math.cos(math.radians(theta_angle))*camera_distance)*math.cos(beta_angle)
    y_camera=(math.cos(math.radians(theta_angle))*camera_distance)*math.sin(beta_angle)
    camera_object.location = (x_camera, y_camera, z_camera) 

    target_location = mathutils.Vector((0, 0, 0))
    direction = target_location - camera_object.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera_object.rotation_euler = rot_quat.to_euler()

    bpy.context.scene.camera = camera_object
    
    if bpy.context.scene.matrix and bpy.context.scene.stereo:
        create_laser_grid("Laser", (x_camera, y_camera, z_camera), bpy.context.scene.ir_side, math.radians(bpy.context.scene.ir_separation), rot_quat.to_euler())
    
    #Generate Render
    scene_renderer = SceneRenderer(context, context.scene.output_directory,scene,i+1)
    scene_renderer.render_all()
    
    if(bpy.context.scene.stereo):
        single_disparity_map(bpy.context.scene.output_directory, scene)
    
    if(bpy.context.scene.gq):
        Generate_GQ(context, bpy.context.scene.output_directory, scene)
            
    if not (bpy.context.scene.segmentation_mode == 'SEMANTIC') and bpy.context.scene.bounding_boxes:
        create_bounding_boxes_from_directory(context.scene.output_directory, scene)    
        
        
    return {'FINISHED'}
        
        
class SceneRenderer:
    def __init__(self, context, output_directory, scene, shot):
        self.context = context
        self.output_directory = output_directory
        self.scene = scene
        self.shot = shot
        self.render_rgb = context.scene.rgb
        self.render_depth = context.scene.depth
        self.render_normals = context.scene.normals
        self.render_segmentation = context.scene.segmentation
        
        if bpy.context.scene.stereo:
            self.setup_stereoscopy()
            bpy.context.scene.render.use_multiview = True
            
        self.context.scene.render.engine = 'CYCLES'
        
        prefs = bpy.context.preferences
        cprefs = prefs.addons['cycles'].preferences
        
        # Update device list
        cprefs.get_devices()

        # Check for CUDA and set it as the compute device type
        if any(device.type == 'CUDA' for device in cprefs.devices):
            cprefs.compute_device_type = 'CUDA'
            bpy.context.scene.cycles.device = 'GPU'

            for device in cprefs.devices:
                if device.type == 'CUDA':
                    device.use = True  # Enable CUDA devices

            print("CUDA device is enabled for rendering.")
        else:
            print("CUDA device not found. Using CPU for rendering.")

    def ensure_directory(self, subfolder):
        path = os.path.join(self.output_directory, subfolder)
        if not os.path.exists(path):
            os.makedirs(path)
            
    def setup_stereoscopy(self):
        self.context.scene.render.use_multiview = True
        self.context.scene.render.views_format = 'STEREO_3D'

    def setup_compositor(self):
        scene = self.context.scene
        scene.use_nodes = True
        tree = scene.node_tree
        tree.nodes.clear()

        render_layers = tree.nodes.new('CompositorNodeRLayers')
        view_layer = self.context.view_layer  

        # Enable necessary render passes
        if(self.render_rgb):
            view_layer.use_pass_combined = True
            self.setup_rgb_output(tree, render_layers)
            
        if(self.render_depth):
            view_layer.use_pass_z = True
            self.setup_depth_output(tree, render_layers)
            
        if(self.render_normals):
            view_layer.use_pass_normal = True
            self.setup_normals_output(tree, render_layers)
            
        if(self.render_segmentation):
            view_layer.use_pass_object_index = True
            self.setup_segmentation_output(tree, render_layers)

    def setup_rgb_output(self, tree, render_layers):
        self.ensure_directory('rgb')
        file_output = tree.nodes.new(type='CompositorNodeOutputFile')
        file_output.base_path = f"{self.output_directory}/rgb/"
        file_output.file_slots[0].path = f"{self.scene}_rgb"
        file_output.format.file_format = 'PNG' 
        tree.links.new(render_layers.outputs['Image'], file_output.inputs[0])

    def setup_depth_output(self, tree, render_layers):
        self.ensure_directory('depth')
        file_output = tree.nodes.new(type='CompositorNodeOutputFile')
        file_output.base_path = f"{self.output_directory}/depth/"
        file_output.file_slots[0].path = f"{self.scene}_depth"  # Change the file extension to '.exr'
        file_output.format.file_format = 'OPEN_EXR'  # Set the file format to OpenEXR
        tree.links.new(render_layers.outputs['Depth'], file_output.inputs[0])


    def setup_normals_output(self, tree, render_layers):
        self.ensure_directory('normals')
        file_output = tree.nodes.new(type='CompositorNodeOutputFile')
        file_output.base_path = f"{self.output_directory}/normals/"
        file_output.file_slots[0].path = f"{self.scene}_normals"
        file_output.format.file_format = 'OPEN_EXR' 
        file_output.format.color_mode = 'RGB'   
        tree.links.new(render_layers.outputs['Normal'], file_output.inputs[0])
      

    def setup_segmentation_output(self, tree, render_layers):
        self.ensure_directory('segmentation')

        # Set up the compositor nodes for segmentation
        scale_node = tree.nodes.new('CompositorNodeMath')
        scale_node.operation = 'DIVIDE'
        scale_node.inputs[1].default_value = 255.0

        tree.links.new(render_layers.outputs['IndexOB'], scale_node.inputs[0])

        file_output_node = tree.nodes.new('CompositorNodeOutputFile')
        file_output_node.base_path = f"{self.output_directory}/segmentation/"
        file_output_node.file_slots[0].path = f"{self.scene}_segmentation"
        
        file_output_node.format.file_format = 'PNG'
        file_output_node.format.color_mode = 'BW'

        tree.links.new(scale_node.outputs[0], file_output_node.inputs[0])

    def render_all(self):
        r = bpy.context.scene.render
        r.resolution_x = bpy.context.scene.res_x
        r.resolution_y = bpy.context.scene.res_y
        self.setup_compositor()
        bpy.ops.render.render(write_still=True)
        remove_numbers_from_filenames(bpy.context.scene.output_directory, self.scene)
        if bpy.context.scene.stereo:
            delete_stereo(self.context.scene.output_directory, self.scene)
            
            
 

        
def remove_numbers_from_filenames(output_directory, start_number):
    prefix = f"{start_number}_"
    for subdir, dirs, files in os.walk(output_directory):
        for filename in files:
            # Filter files that start with the specified number and underscore
            if filename.startswith(prefix):
                if bpy.context.scene.stereo:
                    new_filename = re.sub(r'\d{4}(?=_[LR])', '', filename)
                    if new_filename != filename:
                        old_file_path = os.path.join(subdir, filename)
                        new_file_path = os.path.join(subdir, new_filename)
                        shutil.move(old_file_path, new_file_path)
                else:
                    new_filename = re.sub(r'(?<=\D)\d+(?=\.[^.]+$)', '', filename)
                    if new_filename != filename:
                        old_file_path = os.path.join(subdir, filename)
                        new_file_path = os.path.join(subdir, new_filename)
                        shutil.move(old_file_path, new_file_path)
                    
        
def delete_stereo(output_directory, start_number):
    prefix = f"{start_number}_"
    for subdir, dirs, files in os.walk(output_directory):
        for filename in files:
            if filename.startswith(prefix):
                file_path = os.path.join(subdir, filename)
                if not 'rgb' in filename and re.search(r'_R\.\w+$', filename):
                    os.remove(file_path)
                elif not 'rgb' in filename and re.search(r'_L\.\w+$', filename):
                    new_filename = re.sub(r'_L(\.\w+)$', r'\1', filename)
                    new_file_path = os.path.join(subdir, new_filename)
                    shutil.move(file_path, new_file_path)
                
    
                
                    
def clear_output_directory(output_directory):
    # Iterate through each item in the given directory
    for item_name in os.listdir(output_directory):
        item_path = os.path.join(output_directory, item_name)  # Get the full path of the item
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)  # Remove the file or link
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove the directory and all its contents
        except Exception as e:
            print(f"Error removing {item_path}: {e}")  # Print any exception encountered

            
            
            
def export_to_text_file_segmentation(data, filename, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    full_file_path = os.path.join(folder_path, filename)
    with open(full_file_path, 'w') as file:
        file.write("Name,Index\n")
        for index, name in data:
            file.write(f"{index},{name}\n")
            
            
            
            
def create_bounding_boxes_from_directory(output_directory,scene):
    segmentation_directory = os.path.join(output_directory, "segmentation")
    bounding_box_directory = os.path.join(output_directory, "bounding_box")
    if not os.path.exists(bounding_box_directory):
            os.makedirs(bounding_box_directory)
    
    filename = str(scene) + '_segmentation.png'   
    png_path = os.path.join(segmentation_directory, filename)
    bounding_boxes = create_bounding_boxes_from_png(png_path)
    if bounding_boxes:
        filename = filename.replace("segmentation", "")
        filename = filename.replace(".png", "")
        export_to_text_file_bounding(bounding_boxes, f'{filename}bounding_box.txt', bounding_box_directory)


def create_bounding_boxes_from_png(png_path):
    mask = mpimg.imread(png_path)
    bounding_boxes = []
    if mask.size == 0 or not np.any(mask):
        return bounding_boxes
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids != 0] * 255  
    object_ids = np.round(object_ids).astype(int)  
    for object_id in object_ids:
        rows, cols = np.where(mask == object_id / 255)
        min_x, max_x = np.min(cols), np.max(cols)
        min_y, max_y = np.min(rows), np.max(rows)
        object_name = get_object_name_by_pass_index(object_id)
        bounding_boxes.append([object_id, object_name, min_x, min_y, max_x, max_y])
    return bounding_boxes


def export_to_text_file_bounding(data, filename, folder_path):
    data = [(ObjectID, re.sub(r'\d+$', '', ObjectName), MinX, MinY, MaxX, MaxY) for ObjectID, ObjectName, MinX, MinY, MaxX, MaxY in data]
    full_file_path = os.path.join(folder_path, filename)
    with open(full_file_path, 'w') as file:
        file.write("ObjectID,ObjectName,MinX,MinY,MaxX,MaxY\n")
        for item in data:
            ObjectID, ObjectName, MinX, MinY, MaxX, MaxY = item
            file.write(f"{ObjectID},{ObjectName},{MinX},{MinY},{MaxX},{MaxY}\n")


    
def get_object_name_by_pass_index(pass_index):
    for obj in bpy.context.scene.objects:
        if obj.pass_index == pass_index:
            return obj.name
    return None  

def export_camera_info(output_path):
    camera = bpy.context.scene.camera
    focal_length_mm = camera.data.lens
    sensor_width_mm = camera.data.sensor_width
    location = camera.location
    rotation_euler = camera.rotation_euler
    with open(output_path, 'w') as file:
        file.write(f"Camera Intrinsics:\n")
        file.write(f"Focal Length (mm): {focal_length_mm}\n")
        file.write(f"Sensor Width (mm): {sensor_width_mm}\n\n")
        
        file.write(f"Camera Extrinsics:\n")
        file.write(f"Location (x, y, z) (cm): {location}\n")
        file.write(f"Rotation Euler (x, y, z): {rotation_euler}\n")
        
def export_parameters(output_path):
    with open(output_path, 'w') as file:
        file.write(f"Bin:\n")
        file.write(f"Min Scaling: {bpy.context.scene.min_bin_scaling}\n")
        file.write(f"Max Scaling: {bpy.context.scene.max_bin_scaling}\n")
        file.write(f"Rotation: {bpy.context.scene.bin_rotation}\n\n")
        
        file.write(f"Objects:\n")
        file.write(f"Min Objects: {bpy.context.scene.min_objects}\n")
        file.write(f"Max Objects: {bpy.context.scene.max_objects}\n")
        file.write(f"Scaling: {bpy.context.scene.scaling}\n")
        if bpy.context.scene.scaling:
            file.write(f"Min Scaling: {bpy.context.scene.min_scaling}\n")
            file.write(f"Max Scaling: {bpy.context.scene.min_scaling}\n")
        file.write(f"\n")
        
        
        file.write(f"Lighting:\n")
        file.write(f"Max Sources: {bpy.context.scene.max_lights}\n")
        file.write(f"Max Strenght: {bpy.context.scene.max_light_strength}\n")
        file.write(f"Random Colors: {bpy.context.scene.color_light}\n\n")
        
        file.write(f"Camera:\n")
        file.write(f"Max Angle: {bpy.context.scene.max_camera_angle}\n")
        file.write(f"Min Height: {bpy.context.scene.min_camera}\n")
        file.write(f"Max Height: {bpy.context.scene.max_camera}\n\n")
        
        file.write(f"Resolution x: {bpy.context.scene.res_x}\n")
        file.write(f"Resolution y: {bpy.context.scene.res_y}\n")
        file.write(f"Focal Length: {bpy.context.scene.focal_length}\n")
        file.write(f"Sensor Width: {bpy.context.scene.sensor_width}\n")
        file.write(f"Sensor Height: {bpy.context.scene.sensor_height}\n")
        file.write(f"Stereo Image: {bpy.context.scene.stereo}\n\n")
        
        file.write(f"Output:\n")
        file.write(f"RGB: {bpy.context.scene.rgb}\n")
        file.write(f"Depth Map: {bpy.context.scene.depth}\n")
        file.write(f"Normal Vectors: {bpy.context.scene.normals}\n")
        file.write(f"Segmentation: {bpy.context.scene.segmentation}\n")
        file.write(f"GQ: {bpy.context.scene.gq}\n")
        file.write(f"Dataset Size: {bpy.context.scene.dataset_size}\n\n")
        
        if bpy.context.scene.stereo:
            file.write(f"Stereo:\n")
            file.write(f"Baseline: {bpy.context.scene.camera_baseline}\n")
            file.write(f"IR Matrix: {bpy.context.scene.matrix}\n")
            if bpy.context.scene.matrix:
                file.write(f"Matrix size: {bpy.context.scene.ir_side}\n")
                file.write(f"Dot separation (Degrees): {bpy.context.scene.ir_separation}\n")
            file.write(f"\n")
                
        if bpy.context.scene.gq:
            file.write(f"Grasp Quality:\n")
            
            file.write(f"Distance to Center of Mass: {bpy.context.scene.dist_center}\n")
            if bpy.context.scene.dist_center:
                file.write(f"Distance to Center of Mass Weight: {bpy.context.scene.weight_distance}\n")
                
            file.write(f"Height: {bpy.context.scene.gq_height}\n")
            if bpy.context.scene.gq_height:
                file.write(f"Height Weight: {bpy.context.scene.weight_height}\n")
                
            file.write(f"Flatness: {bpy.context.scene.gq_flatness}\n")
            if bpy.context.scene.gq_flatness:
                file.write(f"Flatness Weight: {bpy.context.scene.weight_flatness}\n")
                file.write(f"Flatness Exponent: {bpy.context.scene.flat_log}\n")
                
            file.write(f"Smoothness: {bpy.context.scene.gq_smoothness}\n")
            if bpy.context.scene.gq_smoothness:
                file.write(f"Smoothness Weight: {bpy.context.scene.weight_smoothness}\n")
                file.write(f"Smoothness Exponent: {bpy.context.scene.smooth_log}\n")
                
            file.write(f"Ray Tracing: {bpy.context.scene.gq_impacts}\n")
            file.write(f"Ray Tracing Mask Mask: {bpy.context.scene.impacts_mask}\n")
            if not(bpy.context.scene.impacts_mask):
                file.write(f"Ray Tracing Weight: {bpy.context.scene.weight_impacts}\n")
                
                
            file.write(f"Edge Detection: {bpy.context.scene.gq_edge}\n")
            file.write(f"Edge Mask: {bpy.context.scene.edge_mask}\n")
            if not(bpy.context.scene.edge_mask):
                file.write(f"Edge Detection Weight: {bpy.context.scene.weight_edge}\n")
            file.write(f"Suction Cup Diameter: {bpy.context.scene.cup_diameter}\n")
            file.write(f"Edge Angle: {bpy.context.scene.edge_angle}\n")
            file.write(f"\n")
            
        if bpy.context.scene.segmentation:
            file.write(f"Segmentation:\n")
            file.write(f"Segmentation Type: {bpy.context.scene.segmentation_mode}\n")
            file.write(f"Bounding Boxes: {bpy.context.scene.bounding_boxes}\n")
            
        




class Generate_GQ:
    def __init__(self, context, output_directory, scene):
        self.context = context
        self.output_directory = output_directory
        depth_directory = os.path.join(self.output_directory, "depth")
        
        gq_path = os.path.join(self.output_directory, "GQ")
        if not os.path.exists(gq_path):
            os.makedirs(gq_path)
        
        filename = str(scene) + '_depth.exr'
        
        points_3d_world, vector_array = self.import_maps(filename, output_directory)
        
        gq = np.zeros(vector_array.shape[:2], dtype=np.float32)
        
        distance, height, flatness, smoothness, impacts, edge = self.object_iteration(filename, self.output_directory, points_3d_world, vector_array)
        
        if context.scene.dist_center:
            gq=gq+distance*context.scene.weight_distance
        if context.scene.gq_height:
            gq=gq+height*context.scene.weight_height
        if context.scene.gq_flatness:
            gq=gq+flatness*context.scene.weight_flatness
        if context.scene.gq_smoothness:
            gq=gq+smoothness*context.scene.weight_smoothness
        if context.scene.gq_impacts:
            if context.scene.impacts_mask:
                gq=gq*impacts
            else:
                gq=gq+impacts*context.scene.weight_impacts
        if context.scene.gq_edge:
            if context.scene.edge_mask:
                gq=gq*edge
            else:
                gq=gq+edge*context.scene.weight_edge
                
        min_gq = np.nanmin(gq)
        max_gq = np.nanmax(gq)
        gq = (gq - min_gq) / (max_gq - min_gq)
                
        new_name = filename.replace('depth', 'GQ').replace('exr', 'csv')
        
        array_no_nans = np.nan_to_num(gq, nan=0)
        np.savetxt(os.path.join(gq_path, new_name), array_no_nans, delimiter=',', fmt='%f')
        
        if bpy.context.scene.positions:
            base_output_dir = bpy.context.scene.output_directory

            specific_positions_dir = os.path.join(base_output_dir, "positions", str(scene))
            os.makedirs(specific_positions_dir, exist_ok=True)

            # Export x, y, z, gq, vx, vy, vz to CSV files in the 'positions' directory
            array_no_nans = np.nan_to_num(points_3d_world[:,:,0], nan=0)
            np.savetxt(os.path.join(specific_positions_dir, "x.csv"), array_no_nans, delimiter=',', fmt='%f')

            array_no_nans = np.nan_to_num(points_3d_world[:,:,1], nan=0)
            np.savetxt(os.path.join(specific_positions_dir, "y.csv"), array_no_nans, delimiter=',', fmt='%f')

            array_no_nans = np.nan_to_num(points_3d_world[:,:,2], nan=0)
            np.savetxt(os.path.join(specific_positions_dir, "z.csv"), array_no_nans, delimiter=',', fmt='%f')

            array_no_nans = np.nan_to_num(vector_array[:,:, 0], nan=0)
            np.savetxt(os.path.join(specific_positions_dir, "Vx.csv"), array_no_nans, delimiter=',', fmt='%f')

            array_no_nans = np.nan_to_num(vector_array[:,:, 1], nan=0)
            np.savetxt(os.path.join(specific_positions_dir, "Vy.csv"), array_no_nans, delimiter=',', fmt='%f')

            array_no_nans = np.nan_to_num(vector_array[:,:, 2], nan=0)
            np.savetxt(os.path.join(specific_positions_dir, "Vz.csv"), array_no_nans, delimiter=',', fmt='%f')
            
        
        if bpy.context.scene.gq and bpy.context.scene.gq_elements:
            specific_elements_dir = os.path.join(base_output_dir, "elements", str(scene))
            os.makedirs(specific_elements_dir, exist_ok=True)
            
            array_no_nans = np.nan_to_num(distance, nan=0)
            np.savetxt(os.path.join(specific_elements_dir, "distance.csv"), array_no_nans, delimiter=',', fmt='%f')
            
            array_no_nans = np.nan_to_num(height, nan=0)
            np.savetxt(os.path.join(specific_elements_dir, "height.csv"), array_no_nans, delimiter=',', fmt='%f')
            
            array_no_nans = np.nan_to_num(flatness, nan=0)
            np.savetxt(os.path.join(specific_elements_dir, "flatness.csv"), array_no_nans, delimiter=',', fmt='%f')
            
            array_no_nans = np.nan_to_num(smoothness, nan=0)
            np.savetxt(os.path.join(specific_elements_dir, "smoothness.csv"), array_no_nans, delimiter=',', fmt='%f')
            
            array_no_nans = np.nan_to_num(impacts, nan=0)
            np.savetxt(os.path.join(specific_elements_dir, "impacts.csv"), array_no_nans, delimiter=',', fmt='%f')
            
            array_no_nans = np.nan_to_num(edge, nan=0)
            np.savetxt(os.path.join(specific_elements_dir, "edge.csv"), array_no_nans, delimiter=',', fmt='%f')
            

                                
                
    @classmethod
    def import_maps(cls, filename, output_directory):
        def read_exr_file(filepath, channel='B'):
            # Check if the original file exists
            if not os.path.exists(filepath):
                # Try modifying the filename to include '3500' right before '.exr'
                modified_filepath = filepath.replace('_depth.exr', '_depth3500.exr')
                if os.path.exists(modified_filepath):
                    filepath = modified_filepath
                else:
                    print(f"File not found: {filepath} and attempted {modified_filepath}")
                    return None
            try:
                exrfile = OpenEXR.InputFile(filepath)
                dw = exrfile.header()['dataWindow']
                size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

                float_pt = Imath.PixelType(Imath.PixelType.FLOAT)
                raw_bytes = exrfile.channel(channel, float_pt)
                data_vector = np.frombuffer(raw_bytes, dtype=np.float32)
                data_map = np.reshape(data_vector, (size[1], size[0]))
                return data_map
            except Exception as e:
                print(f"Failed to read or process EXR file {filepath}: {str(e)}")
                return None

        # Read depth map
        depth_map_path = os.path.join(output_directory, 'depth', filename)
        depth_array = read_exr_file(depth_map_path, 'B')
        if depth_array is None:
            print("Error: Failed to load depth data.")
            return None, None

        # Read vector map
        vector_map_path = os.path.join(output_directory, 'normals', filename.replace('depth', 'normals'))
        vector_array_r = read_exr_file(vector_map_path, 'R')
        vector_array_g = read_exr_file(vector_map_path, 'G')
        vector_array_b = read_exr_file(vector_map_path, 'B')
        vector_array = np.stack((vector_array_r, -vector_array_g, vector_array_b), axis=-1)  # Invert Y-axis for vector

        camera = bpy.context.scene.camera
        if not camera:
            print("Error: No camera found.")
            return None, None

        # Camera parameters
        focal_length_mm = camera.data.lens
        sensor_width_mm = camera.data.sensor_width
        image_width_px = bpy.context.scene.render.resolution_x
        image_height_px = bpy.context.scene.render.resolution_y
        cx = image_width_px / 2
        cy = image_height_px / 2
        px_size_mm = sensor_width_mm / image_width_px

        # Compute 3D points
        points_3d_world = np.zeros((image_height_px, image_width_px, 3), dtype=np.float32)
        matrix = camera.matrix_world
        location, rotation, _ = matrix.decompose()
        rotation_matrix_np = np.array(rotation.to_matrix().to_4x4())[0:3, 0:3]
        translation_vector = np.array(location)

        for y in range(image_height_px):
            for x in range(image_width_px):
                Z_cm = -depth_array[y, x]
                X_cm = -((x - cx) * Z_cm * px_size_mm) / focal_length_mm
                Y_cm = ((y - cy) * Z_cm * px_size_mm) / focal_length_mm  
                point_camera_cm = np.array([X_cm, Y_cm, Z_cm])
                points_3d_world[y, x] = np.dot(rotation_matrix_np, point_camera_cm) + translation_vector



        norms = np.linalg.norm(vector_array, axis=2, keepdims=True)
        vector_array = vector_array / norms

            
        return points_3d_world, vector_array
    
    
    @classmethod
    def object_iteration(cls, filename, output_directory, points_3d_world, normal_vectors):

        # Import segmentation mask and normalize
        filename_segmentation = filename.replace('depth', 'segmentation').replace('exr', 'png')
        segmentation_mask_path = os.path.join(output_directory, 'segmentation', filename_segmentation)
        segmentation_mask = cv2.imread(segmentation_mask_path, cv2.IMREAD_UNCHANGED)
        if segmentation_mask is None:
            raise FileNotFoundError(f"Cannot read segmentation mask from path: {segmentation_mask_path}")
        pass_index_mask = segmentation_mask.astype(int)
        
        # Obtain object indexes
        objects_data = []
        for obj in bpy.context.scene.objects:
            pass_index = obj.pass_index
            if pass_index != 0:
                objects_data.append({'Object': obj.name, 'PassIndex': pass_index})

        objects = pd.DataFrame(objects_data)
        objects.sort_values(by='PassIndex', inplace=True)
        objects = objects.iloc[1:]
        objects.reset_index(drop=True, inplace=True)
        
        
        # Initialize an arrays
        distances_to_center = np.full(pass_index_mask.shape, np.nan, dtype=float)
        normalized_distances_to_center = np.full(pass_index_mask.shape, np.nan, dtype=float)
        height = np.full(pass_index_mask.shape, np.nan, dtype=float)
        flatness = np.full(pass_index_mask.shape, np.nan, dtype=float)
        smoothness = np.full(pass_index_mask.shape, np.nan, dtype=float)
        impacts = np.full(pass_index_mask.shape, np.nan, dtype=float)
        edge = np.full(pass_index_mask.shape, np.nan, dtype=float)
    
        for index, row in objects.iterrows():
            obj = bpy.context.scene.objects.get(row['Object'])
            if obj:
                
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)
                center_of_mass = np.array(obj.matrix_world.translation)
                
                pass_index = row['PassIndex']
                binary_map = (pass_index_mask == pass_index)
            
                indexes, normalized_masked_values = cls.distance_to_center(binary_map, points_3d_world, center_of_mass, distances_to_center)
                distances_to_center[indexes] = normalized_masked_values
                
                flatness_complete, smoothness_complete = cls.flatness(points_3d_world, normal_vectors, binary_map)
                flatness[binary_map] = np.where(np.isnan(flatness_complete[binary_map]), flatness[binary_map], flatness_complete[binary_map])
                smoothness[binary_map] = np.where(np.isnan(smoothness_complete[binary_map]), smoothness[binary_map], smoothness_complete[binary_map])
                
                edge_detection_complete = cls.edge_detection(points_3d_world, normal_vectors, binary_map)
                edge[binary_map] = edge_detection_complete[binary_map]

                
        impact_dfs = []  # Initialize a list to store impact DataFrames
        
        for x in range(segmentation_mask.shape[0]):
            for y in range(segmentation_mask.shape[1]):
                index = segmentation_mask[x, y]
                if index > 1:
                    object_name = objects.loc[objects['PassIndex'] == index, 'Object'].iloc[0]
                    obj = bpy.context.scene.objects.get(object_name)
                    point_3d = points_3d_world[x, y, :]
                    normal_vec = normal_vectors[x, y, :]

                    height[x, y] = points_3d_world[x, y, 2]
                    impacts[x,y], impact_df = cls.impacts(point_3d, normal_vec, obj)
                    if not impacts[x,y]:
                        impact_dfs.append(impact_df)  # Append impact DataFrame to list

        # Concatenate all impact DataFrames into a single DataFrame
        if impact_dfs:
            impacts_df = pd.concat(impact_dfs)
            #impacts_df.to_csv(os.path.join('C:/Users/andre/Desktop', 'impacts123.csv'), index=False)
    
        min_height = np.nanmin(height)
        max_height = np.nanmax(height)
        normalized_height = (height - min_height) / (max_height - min_height)
        
        
        flatness_exp= bpy.context.scene.flat_log
        flatness = (1-flatness)**flatness_exp
        min_flatness = np.nanmin(flatness)
        max_flatness = np.nanmax(flatness)
        normalized_flatness = (flatness - min_flatness) / (max_flatness - min_flatness)
        
        smoothness_exp= bpy.context.scene.smooth_log
        smoothness = (1-smoothness)**smoothness_exp
        min_smoothness = np.nanmin(smoothness)
        max_smoothness = np.nanmax(smoothness)
        normalized_smoothness = (smoothness - min_smoothness) / (max_smoothness - min_smoothness)
        
        
        return distances_to_center, normalized_height, normalized_flatness, normalized_smoothness, impacts, edge
                    
                    
    @classmethod
    def distance_to_center(cls, binary_map, points_3d_world, center_of_mass, distances_to_center):
        for x in range(binary_map.shape[0]):
            for y in range(binary_map.shape[1]):
                if binary_map[x, y]:
                    point_3d = points_3d_world[x, y, :]
                    
                    # Correct the distance calculation
                    dx = point_3d[0] - center_of_mass[0]
                    dy = point_3d[1] - center_of_mass[1]
                    dz = point_3d[2] - center_of_mass[2]
                    distance = np.sqrt(dx**2 + dy**2 + dz**2)
                    
                    distances_to_center[x, y] = distance
        
        # Check if the array is empty before trying to find max and min
        indexes = np.where(binary_map == 1)
        masked_values = distances_to_center[indexes]
        
        if masked_values.size > 0:
            max_value = masked_values.max()
            min_value = masked_values.min()
            # Avoid division by zero in case max_value equals min_value
            if max_value != min_value:
                normalized_masked_values = 1 - ((masked_values - min_value) / (max_value - min_value))
            else:
                # Handle the case where all distances are the same
                normalized_masked_values = np.zeros_like(masked_values)
        else:
            # Handle the case where the array is empty
            normalized_masked_values = np.array([])  # or an appropriate default value
        
        return indexes, normalized_masked_values
    
    

    @classmethod
    def flatness(cls, points_3d_world, normal_vectors, binary_map, kernel_size=4):
        pad_size = kernel_size // 3
        height, width, _ = normal_vectors.shape
        
        # Initialize an output array for normal differences and smoothness
        normal_difference_image = np.zeros((height, width), dtype=np.float32)
        smoothness_difference_image = np.zeros((height, width), dtype=np.float32)
        
        # Pad the normal_vectors and points_3d_world arrays to handle edges
        padded_normals = np.pad(normal_vectors, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')
        padded_points = np.pad(points_3d_world, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')
        
        for y in range(height):
            for x in range(width):
                if binary_map[y, x]:
                    # Extract the neighborhood of normals
                    neighborhood_normals = padded_normals[y:y + kernel_size, x:x + kernel_size, :].reshape(-1, 3)
                    
                    # Extract the neighborhood of points with correct padding consideration
                    neighborhood_points = padded_points[y:y + kernel_size, x:x + kernel_size, :].reshape(-1, 3)
                    
                    # Central normal vector and point
                    central_normal = normal_vectors[y, x, :]
                    P0 = points_3d_world[y, x]
                    
                    # FLATNESS
                    # Calculate the difference in x and y components between the central normal and each neighbor's normal
                    differences_xy = np.sqrt(np.sum((neighborhood_normals[:, :2] - central_normal[:2])**2, axis=1))
                    
                    # Summarize the differences to quantify the variation
                    normal_difference_image[y, x] = np.mean(differences_xy)
                    
                    # SMOOTHNESS calculation: Adjust to calculate mean squared distance to the fitted plane
                
                    # Fit a plane to the neighborhood points using SVD
                    centroid = np.mean(neighborhood_points, axis=0)
                    centered_points = neighborhood_points - centroid
                    U, S, Vt = np.linalg.svd(centered_points, full_matrices=False)
                    plane_normal = Vt[-1, :]
                    
                    # Calculate 'd' for the plane equation
                    d = -np.dot(plane_normal, centroid)
                    
                    # Calculate squared distances of points to the plane
                    distances_to_plane = np.abs(np.dot(neighborhood_points, plane_normal) + d) / np.linalg.norm(plane_normal)
                    squared_distances = distances_to_plane**2
                    
                    # Store the mean squared distance as a measure of smoothness
                    smoothness_difference_image[y, x] = np.mean(squared_distances)
        
        return normal_difference_image, smoothness_difference_image


                    
    
    @classmethod
    def impacts(cls, point_3d, normal_vec, obj):
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()        
        ray_origin = point_3d
        ray_direction = normal_vec
        result, location, normal, index, hit_obj, matrix = bpy.context.scene.ray_cast(depsgraph, ray_origin, ray_direction)
        
        impact_data = []
        impact_data.append({
            'Origin': point_3d,
            'Impacts': True,
            'Impacted_Object': hit_obj.name if hit_obj else None,
            'Origin_Object': obj.name
        })

        impact_df = pd.DataFrame(impact_data)
        
        if result and hit_obj != obj:
            return False, impact_df
        else:
            return True, impact_df


    @classmethod
    def edge_detection(cls, points_3d_world, normal_vectors, binary_map, kernel_size=3, angle_threshold=30, suction_cup_radius=1):
        pad_size = kernel_size // 3
        height, width, _ = normal_vectors.shape

        # Access the user-defined properties for suction cup radius and angle threshold
        suction_cup_radius = bpy.context.scene.cup_diameter / 20
        angle_threshold = bpy.context.scene.edge_angle

        # Access the user-defined properties for resolution
        res_x = bpy.context.scene.res_x
        res_y = bpy.context.scene.res_y

        # Initialize the edge detection image as False (no edge)
        edge_detection_image = np.zeros((height, width), dtype=bool)

        # Create a kernel to check for neighbors
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.uint8)

        # Convolve the binary map with the kernel to identify the object edges
        object_edges = convolve(binary_map.astype(np.uint8), kernel, mode='constant', cval=0) < kernel.sum()

        # Mark the object edges on the edge detection image
        edge_detection_image |= object_edges

        # Define a function to calculate the angle between two normals
        def calculate_angle(normal_a, normal_b):
            """Calculate the angle between two normals."""
            cos_angle = np.dot(normal_a, normal_b) / (np.linalg.norm(normal_a) * np.linalg.norm(normal_b))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp to handle numerical instability
            return degrees(acos(cos_angle))

        # Pad the normal_vectors to handle edges
        padded_normals = np.pad(normal_vectors, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')

        # Iterate over each point in the binary map
        for y in range(height):
            for x in range(width):
                if binary_map[y, x]:
                    # Extract the neighborhood of normals directly using the padded array
                    neighborhood_normals = padded_normals[y:y + kernel_size, x:x + kernel_size, :].reshape(-1, 3)

                    # Central normal vector
                    central_normal = padded_normals[y + pad_size, x + pad_size]

                    # Calculate angles between the central normal and its neighbors
                    for neighbor_normal in neighborhood_normals:
                        # Skip the comparison with the central normal itself
                        if np.array_equal(central_normal, neighbor_normal):
                            continue

                        angle = calculate_angle(central_normal, neighbor_normal)
                        if angle > angle_threshold:
                            edge_detection_image[y, x] = True
                            break  # If one neighbor exceeds the threshold, mark as edge

        # Camera properties for scaling
        camera = bpy.context.scene.camera
        focal_length_mm = camera.data.lens  # Focal length in mm
        sensor_width_mm = camera.data.sensor_width  # Sensor width in mm

        # Calculate the focal length in pixels for each axis
        focal_length_px_x = (focal_length_mm / sensor_width_mm) * bpy.context.scene.res_x
        focal_length_px_y = (focal_length_mm / sensor_width_mm) * bpy.context.scene.res_y

        # Average depth from the points_3d_world data
        average_depth = np.mean(points_3d_world[:,:,2])

        # Calculate the pixel scale for each axis
        pixel_scale_x = focal_length_px_x / (average_depth * 10)
        pixel_scale_y = focal_length_px_y / (average_depth * 10)

        # Use the average pixel scale for the suction cup radius calculation
        pixel_scale = (pixel_scale_x + pixel_scale_y) / 2
        radius_in_pixels = int(round(suction_cup_radius * pixel_scale))

        dilation_structure = disk(radius_in_pixels)
        expanded_edges = binary_dilation(edge_detection_image, dilation_structure)

        expanded_edges = 1 - expanded_edges

        return expanded_edges
    

                    

def single_disparity_map(output_directory, scene):
    def generate_disparity_map(left_image, right_image, params):
        """ Generate a disparity map using StereoSGBM and given parameters. """
        stereo = cv2.StereoSGBM_create(
            minDisparity=params['minDisparity'],
            numDisparities=params['numDisparities'],
            blockSize=params['blockSize'],
            P1=params['P1'],
            P2=params['P2'],
            disp12MaxDiff=params['disp12MaxDiff'],
            uniquenessRatio=params['uniquenessRatio'],
            speckleWindowSize=params['speckleWindowSize'],
            speckleRange=params['speckleRange'],
            preFilterCap=params['preFilterCap'],
            mode=params['mode']
        )
        disparity_map = stereo.compute(left_image, right_image).astype(np.float32) / 16.0
        return disparity_map

    def calculate_laplacian_variance(disparity_map):
        """ Calculate the variance of the Laplacian (edge detection) response of the disparity map. """
        disparity_map_32f = disparity_map.astype(np.float32)
        laplacian = cv2.Laplacian(disparity_map_32f, cv2.CV_32F)
        return laplacian.var()

    def calculate_tv_norm(disparity_map):
        """ Calculate the Total Variation norm of the disparity map. """
        tv_norm = np.sum(np.abs(np.gradient(disparity_map)))
        return tv_norm

    def normalize(values):
        """ Normalize the values to the range [0, 1]. """
        min_val = np.min(values)
        max_val = np.max(values)
        normalized_values = (values - min_val) / (max_val - min_val)
        return normalized_values

    save_path = os.path.join(output_directory, 'stereo_depth_map')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    focal_length = bpy.context.scene.focal_length
    baseline = bpy.context.scene.camera_baseline / 100     
    
    directory_path = os.path.join(output_directory, 'rgb')
    
    left_image_path = os.path.join(directory_path, str(scene) + '_rgb_L.png')
    right_image_path = os.path.join(directory_path, str(scene) + '_rgb_R.png')

    # Load left and right images
    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    # Preprocessing
    left_image = cv2.equalizeHist(left_image)
    right_image = cv2.equalizeHist(right_image)
    
    # Define parameter sets
    param_sets = [
        {'minDisparity': 0, 'numDisparities': 160, 'blockSize': 5, 'P1': 50, 'P2': 4000, 'disp12MaxDiff': 25, 'uniquenessRatio': 0, 'speckleWindowSize': 0, 'speckleRange': 0, 'preFilterCap': 1, 'mode': 0},
        {'minDisparity': 0, 'numDisparities': 112, 'blockSize': 3, 'P1': 100, 'P2': 720, 'disp12MaxDiff': 50, 'uniquenessRatio': 0, 'speckleWindowSize': 0, 'speckleRange': 1, 'preFilterCap': 20, 'mode': 0},
        {'minDisparity': -5, 'numDisparities': 96, 'blockSize': 3, 'P1': 50, 'P2': 10480, 'disp12MaxDiff': 31, 'uniquenessRatio': 0, 'speckleWindowSize': 0, 'speckleRange': 0, 'preFilterCap': 43, 'mode': 0},
        {'minDisparity': 0, 'numDisparities': 64, 'blockSize': 3, 'P1': 0, 'P2': 10000, 'disp12MaxDiff': 524, 'uniquenessRatio': 0, 'speckleWindowSize': 0, 'speckleRange': 0, 'preFilterCap': 1, 'mode': 0},
        {'minDisparity': -2, 'numDisparities': 128, 'blockSize': 7, 'P1': 500, 'P2': 6000, 'disp12MaxDiff': 20, 'uniquenessRatio': 5, 'speckleWindowSize': 10, 'speckleRange': 2, 'preFilterCap': 15, 'mode': 0}
    ]

    best_params = None
    min_score = float('inf')
    laplacian_variances = []
    tv_norms = []
    scores = []
    best_disparity_map = None

    for params in param_sets:
        disparity_map = generate_disparity_map(left_image, right_image, params)
        laplacian_variance = calculate_laplacian_variance(disparity_map)
        tv_norm = calculate_tv_norm(disparity_map)
        laplacian_variances.append(laplacian_variance)
        tv_norms.append(tv_norm)

    normalized_laplacian_variances = normalize(laplacian_variances)
    normalized_tv_norms = normalize(tv_norms)

    for idx in range(len(param_sets)):
        score = 0.75 * normalized_laplacian_variances[idx] + 0.25 * normalized_tv_norms[idx]  # Combined metric
        scores.append(score)

        if score < min_score:
            min_score = score
            best_params = param_sets[idx]
            best_disparity_map = generate_disparity_map(left_image, right_image, param_sets[idx])

        print(f"Params {idx}: Laplacian Variance={laplacian_variances[idx]}, TV Norm={tv_norms[idx]}, Score={score}")

    # Postprocessing the best disparity map
    best_disparity_map = cv2.GaussianBlur(best_disparity_map, (5, 5), 0)
    best_disparity_map[best_disparity_map <= 0] = 0.1

    # Calculate the depth map
    focal_length_pixels = (focal_length / bpy.context.scene.sensor_width) * bpy.context.scene.res_x
    depth_map = (focal_length_pixels * baseline) / best_disparity_map
    depth_map[depth_map > 5] = 0

    # Save the depth map as a CSV file
    output_filename = os.path.join(save_path, (str(scene) +'_stereo_depth.csv'))
    with open(output_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in depth_map:
            writer.writerow(row)

    print(f"Best parameters: {best_params}")
    print(f"Saved best depth map to {output_filename}")             
                    



                    

    
        
    
class GenerateDataset(bpy.types.Operator):
    """Generate the dataset"""
    bl_idname = "object.generate_dataset" 
    bl_label = "Generate Dataset"

    def execute(self, context):
        objects_df = fill_dataframe_from_directory(context.scene.object_quality_directory)
        bins_df = fill_dataframe_from_directory(context.scene.bin_quality_directory)
        
        if bpy.context.scene.starting_scene == 1:
            clear_output_directory(context.scene.output_directory)
        
        dataset_size=context.scene.dataset_size
        starting_scene = bpy.context.scene.starting_scene
        for scene in range(starting_scene, starting_scene + dataset_size):
            start_time = datetime.now()
            attempt = 0
            retries = 10
            while attempt < retries:
                try:
                    # Attempt to generate the scene
                    generate_scene(objects_df, bins_df, context, scene)
                    # If successful, break out of the loop
                    break
                except Exception as e:
                    attempt += 1
                    print(f"Error generating scene {scene}: {str(e)}")
                    if attempt >= retries:
                        # If we reach the maximum number of retries, raise the error
                        print(f"Failed after {retries} attempts to generate scene {scene}")
                        raise e
                    else:
                        print(f"Retrying... attempt {attempt + 1}")
            end_time = datetime.now()
            time_difference = end_time - start_time
            print(f"Generar la escena {scene} tomo {time_difference}")

        camera_path=os.path.join(context.scene.output_directory, "camera_data.txt")
        export_camera_info(camera_path)
        
        dataset_path=os.path.join(context.scene.output_directory, "dataset_info.txt")
        export_parameters(dataset_path)
        
        if context.scene.segmentation_mode == 'SEMANTIC':
            data = [('bin',1)]
            objects_reduced = objects_df.drop(['type'], axis=1)
            objects_reduced['Index'] = range(2, 2 + len(objects_reduced))
            objects_reduced['name'] = objects_reduced['name'].str.split('.', n=1).str[0]
            new_data = list(objects_reduced.itertuples(index=False, name=None))
            data += new_data
            export_to_text_file_segmentation(data, 'Index.txt', f"{context.scene.output_directory}/segmentation/")   
                
                
        return {'FINISHED'}
        

        


        
        
# ------------------------------------ REGISTER ----------------------------------------------------------------        

def register():
    
    #PANELS
    bpy.utils.register_class(UI_PT_main)
    bpy.utils.register_class(MODEL_PT_Panel)
    bpy.utils.register_class(PARAMETERS_PT_Panel)
    bpy.utils.register_class(OUTPUT_PT_Panel)
    bpy.utils.register_class(STEREO_PT_Panel)
    bpy.utils.register_class(GQ_PT_Panel)
    bpy.utils.register_class(SEGMENTATION_PT_Panel)
    
    
    bpy.types.Scene.materials_directory = bpy.props.StringProperty(
        name="Object Quality Directory",
        subtype='FILE_PATH',
        description="Path to the directory containing object quality files",
        default="C:/Users/andre/Documents/Blender Materials/materials_data_extended.json"
    )

    
    bpy.utils.register_class(FolderFinder)
    bpy.types.Scene.object_quality_directory = bpy.props.StringProperty(
        name="Object Quality Directory",
        subtype='DIR_PATH',
        description="Path to the directory containing object quality files",
        default="C:/Users/andre/Documents/Recursos Tesis/Objetos_reducido/"
    )
    
    bpy.types.Scene.output_directory = bpy.props.StringProperty(
        name="Output Directory",
        subtype='DIR_PATH',
        description="Path to the directory where output files will be saved",
        default="C:/Users/andre/Documents/Base de datos 15000/"
    )
    
    

    
    #Properties
    bpy.types.Scene.gq_elements = bpy.props.BoolProperty(
        name="gq elements",
        description="Minimum amount of objects per scene",
        default=True
    )
    
    bpy.types.Scene.positions = bpy.props.BoolProperty(
        name="gq elements",
        description="Minimum amount of objects per scene",
        default=True
    )
    
    bpy.types.Scene.min_objects = bpy.props.IntProperty(
        name="Min Objects",
        description="Minimum amount of objects per scene",
        default=1,
        min=1,
        max=70,
        update=update_min_objects
    )
    
    bpy.types.Scene.max_objects = bpy.props.IntProperty(
        name="Max Objects",
        description="Maximum amount of objects per scene",
        default=15,
        min=1,
        max=70,
        update=update_max_objects
    )
    
    bpy.types.Scene.min_bin_scaling = bpy.props.FloatProperty(
        name="Min Bin scaling",
        description="Min Bin scaling",
        default=0.75,
        min=0.25,
        max=2.0,
        update=update_min_bin_scaling
    )
    
    bpy.types.Scene.max_bin_scaling = bpy.props.FloatProperty(
        name="Max Bin scaling",
        description="Max Bin scaling",
        default=1.25,
        min=0.25,
        max=2.0,
        update=update_max_bin_scaling
    )
    
    
    bpy.types.Scene.scaling = bpy.props.BoolProperty(
        name="Scaling",
        description="Show or hide the input boxes",
        default=True
    )
    
    bpy.types.Scene.min_scaling = bpy.props.FloatProperty(
        name="Min Scaling",
        description="Minimum amount of scaling",
        default=0.5,
        min=0.5,
        max=2.0,
        update=update_min_scaling
    )
    
    bpy.types.Scene.max_scaling = bpy.props.FloatProperty(
        name="Max Scaling",
        description="Maximum amount of scaling",
        default=1.5,
        min=0.5,
        max=2.0,
        update=update_max_scaling
    )
    
    bpy.types.Scene.max_lights = bpy.props.IntProperty(
        name="Max lights",
        description="Maximum amount of lights",
        default=3,
        min=0,
        max=5,
    )
    
    bpy.types.Scene.max_light_strength = bpy.props.FloatProperty(
        name="Max Light Strength (%)",
        description="Maximum light strength as a percentage (1% to 100%)",
        default=100.0,  
        min=1.0,        
        max=100.0,      
        subtype='PERCENTAGE'
    )
    
    bpy.types.Scene.max_camera_angle = bpy.props.IntProperty(
        name="Max Camera Angle",
        description="Maximum camera angle",
        default=45,
        min=0,
        max=80,
    )
    
    bpy.types.Scene.min_camera = bpy.props.FloatProperty(
        name="Min Camera height",
        description="Minimum camera height",
        default=1,
        min=0.3,
        max=5.0,
        update=update_min_camera
    )
    
    bpy.types.Scene.max_camera = bpy.props.FloatProperty(
        name="Max Camera height",
        description="Maximum camera height",
        default=2,
        min=0.3,
        max=5.0,
        update=update_max_camera  
    )
    
    bpy.types.Scene.rgb = bpy.props.BoolProperty(
        name="RGB",
        description="RGB",
        default=True
    )
    
    bpy.types.Scene.depth = bpy.props.BoolProperty(
        name="Depth",
        description="Depth",
        default=True
    )
    
    bpy.types.Scene.normals = bpy.props.BoolProperty(
        name="Normals",
        description="Normals",
        default=True
    )
    
    bpy.types.Scene.segmentation = bpy.props.BoolProperty(
        name="Segmentation",
        description="Segmentation",
        default=True
    )
    
    bpy.types.Scene.bounding_boxes = bpy.props.BoolProperty(
        name="Bounding",
        description="Bounding",
        default=True
    )
    
    bpy.types.Scene.dist_center = bpy.props.BoolProperty(
        name="Distance to Center of Mass",
        description="Calculate distance to center of mass",
        default=True
    )
    
    bpy.types.Scene.gq = bpy.props.BoolProperty(
        name="Grasp Quality",
        description="Grasp Quality",
        default=True
    )
    
    bpy.types.Scene.color_light = bpy.props.BoolProperty(
        name="Random color light",
        description="Random Color light",
        default=True
    )
    
    bpy.types.Scene.dataset_size = bpy.props.IntProperty(
        name="Dataset size",
        description="Dataset_size",
        default=500,
        min=1,
        max=20000,
    )
    
    
    bpy.types.Scene.segmentation_mode = bpy.props.EnumProperty(
        name="Segmentation Mode",
        description="Choose the segmentation mode",
        items=segmentation_modes_callback,
    )

    
    bpy.types.Scene.bin_quality_directory = bpy.props.StringProperty(
        name="Bin Quality Directory",
        subtype='DIR_PATH',
        default="C:/Users/andre/Documents/Recursos Tesis/Bins_reducido/",
        description="Path to the directory containing bin quality files"
    )
    
    bpy.types.Scene.bin_rotation = bpy.props.BoolProperty(
        name="Bin Rotation",
        description="Bin Rotation",
        default=True
    )
    
    bpy.types.Scene.gq_height = bpy.props.BoolProperty(
        name="Grasp Quality Height",
        description="Grasp Quality Height",
        default=True
    )
    
    bpy.types.Scene.gq_flatness = bpy.props.BoolProperty(
        name="Grasp Quality Flatness",
        description="Grasp Quality Flatness",
        default=True
    )
    
    bpy.types.Scene.gq_smoothness = bpy.props.BoolProperty(
        name="Grasp Quality Smoothness",
        description="Grasp Quality Smoothness",
        default=True
    )
    
    bpy.types.Scene.gq_impacts = bpy.props.BoolProperty(
        name="Grasp Quality Impacts",
        description="Grasp Quality Impacts",
        default=True
    )
    
    bpy.types.Scene.gq_edge = bpy.props.BoolProperty(
        name="Grasp Quality Edge",
        description="Grasp Quality Edge",
        default=True
    )
    
    bpy.types.Scene.impacts_mask = bpy.props.BoolProperty(
        name="Mask for impacts",
        description="Mask for impacts",
        default=False,
        update=update_impacts_weight
    )
    
    bpy.types.Scene.impacts_weight_bool = bpy.props.BoolProperty(
        name="Weights for impacts",
        description="Weights for impacts",
        default=True,
        update=update_impacts_mask
    )
    
    bpy.types.Scene.edge_mask = bpy.props.BoolProperty(
        name="Mask for edge",
        description="Mask for edge",
        default=False,
        update=update_edge_weight
    )
    
    bpy.types.Scene.edge_weight_bool = bpy.props.BoolProperty(
        name="Weights for edge",
        description="Weights for edge",
        default=True,
        update=update_edge_mask
    )
    
    bpy.types.Scene.weight_height = bpy.props.FloatProperty(
        name="Weight for the height",
        description="Weight for the height",
        default=0.0,
        min=0.0,
        max=50.0, 
    )
    
    bpy.types.Scene.weight_flatness = bpy.props.FloatProperty(
        name="Weight for the flatness",
        description="Weight for the flatness",
        default=1.0,
        min=0.0,
        max=50.0,
    )
    
    bpy.types.Scene.weight_distance = bpy.props.FloatProperty(
        name="Weight for the distance to center of mass",
        description="Weight for the distance to center of mass",
        default=0.5,
        min=0.0,
        max=50.0, 
    )
    
    bpy.types.Scene.weight_smoothness = bpy.props.FloatProperty(
        name="Weight for smoothness",
        description="Weight for smoothness",
        default=1.0,
        min=0.0,
        max=50.0, 
    )
    
    bpy.types.Scene.weight_impacts = bpy.props.FloatProperty(
        name="Weight for imapct",
        description="Weight for impacts",
        default=0.0,
        min=0.0,
        max=50.0, 
    )
    
    bpy.types.Scene.weight_edge = bpy.props.FloatProperty(
        name="Weight for edge",
        description="Weight for edge",
        default=0.5,
        min=0.0,
        max=50.0, 
    )
    
    bpy.types.Scene.cup_diameter = bpy.props.FloatProperty(
        name="Suction Cup Diameter",
        description="Suction Cup Diameter",
        default=50,
        min=1.0,
        max=100.0, 
    )
    
    bpy.types.Scene.flat_log = bpy.props.IntProperty(
        name="Flat log",
        description="Flat log",
        default=8,
        min=1,
        max=10, 
    )
    
    bpy.types.Scene.smooth_log = bpy.props.IntProperty(
        name="Smooth log",
        description="Smooth log",
        default=8,
        min=1,
        max=10, 
    )
    
    bpy.types.Scene.edge_angle = bpy.props.IntProperty(
        name="Edge_angle",
        description="Edge_angle",
        default=20,
        min=15,
        max=60, 
    )
        
    bpy.types.Scene.res_x = bpy.props.IntProperty(
        name="Resolution_x",
        description="Resolution_x",
        default=1280,
        min=640,
        max=3000, 
    )
    
    bpy.types.Scene.res_y = bpy.props.IntProperty(
        name="Resolution_y",
        description="Resolution_y",
        default=720,
        min=640,
        max=3000, 
    )
    bpy.types.Scene.focal_length = bpy.props.FloatProperty(
        name="Focal Lenght",
        description="Focal Lenght",
        default=1.88,
        min=1.0,
        max=50.0, 
    )
    bpy.types.Scene.sensor_width = bpy.props.FloatProperty(
        name="sensor_width",
        description="sensor_width",
        default=3.68,
        min=1.0,
        max=50.0, 
    )
    bpy.types.Scene.sensor_height = bpy.props.FloatProperty(
        name="sensor_height",
        description="sensor_height",
        default=2.76,
        min=1.0,
        max=50.0, 
    )
    
    bpy.types.Scene.camera_baseline = bpy.props.FloatProperty(
        name="camera_baseline",
        description="camera_baseline",
        default=9.5,
        min=1.0,
        max=50.0, 
    )
    
    bpy.types.Scene.ir_side = bpy.props.IntProperty(
        name="ir_side",
        description="ir_side",
        default=10,
        min=3,
        max=20, 
    )
    
    bpy.types.Scene.ir_separation = bpy.props.IntProperty(
        name="ir_separation",
        description="ir_separation",
        default=7,
        min=2,
        max=20, 
    )
    
    bpy.types.Scene.stereo = bpy.props.BoolProperty(
        name="Stereo",
        description="Stereo",
        default=True,
    )
    
    bpy.types.Scene.matrix = bpy.props.BoolProperty(
        name="Stereo matrix",
        description="Stereo matrix",
        default=True,
    )
    
    bpy.types.Scene.starting_scene = bpy.props.IntProperty(
        name="Starting Scene",
        description="Starting scene number",
        default=2,
        min=1,
        max=15000, 
    )
    
    
    
    bpy.utils.register_class(GenerateDataset)
    
    
    
    
def unregister():
    # Unregister all custom panels
    bpy.utils.unregister_class(UI_PT_main)
    bpy.utils.unregister_class(MODEL_PT_Panel)
    bpy.utils.unregister_class(PARAMETERS_PT_Panel)
    bpy.utils.unregister_class(OUTPUT_PT_Panel)
    bpy.utils.unregister_class(STEREO_PT_Panel)  # This was missing
    bpy.utils.unregister_class(GQ_PT_Panel)
    bpy.utils.unregister_class(SEGMENTATION_PT_Panel)
    bpy.utils.unregister_class(FolderFinder)
    bpy.utils.unregister_class(GenerateDataset)
    
    # Remove custom properties added to bpy.types.Scene
    props = [
        "gq_elements","positions","object_quality_directory", "output_directory","materials_directory", "min_objects", "max_objects",
        "min_bin_scaling", "max_bin_scaling", "scaling", "min_scaling", "max_scaling",
        "max_lights", "max_light_strength", "max_camera_angle", "min_camera",
        "max_camera", "renders_scene", "rgb", "depth", "normals", "segmentation",
        "bounding_boxes", "dist_center", "gq", "color_light", "dataset_size",
        "segmentation_mode", "bin_quality_directory", "bin_rotation", "gq_height",
        "gq_flatness", "gq_smoothness", "gq_impacts", "gq_edge", "impacts_mask",
        "impacts_weight_bool", "edge_mask", "edge_weight_bool", "weight_height",
        "weight_flatness", "weight_distance", "weight_smoothness", "weight_impacts",
        "weight_edge", "cup_diameter", "flat_log", "smooth_log", "edge_angle",
        "res_x", "res_y", "focal_length", "sensor_width", "sensor_height",
        "camera_baseline", "ir_side", "ir_separation", "stereo", "matrix",
        "starting_scene"
    ]
    
    for prop in props:
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)




if __name__ == "__main__":
    register()