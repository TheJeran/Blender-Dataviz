import bpy 
import numpy as np
import os
from glob import glob
from math import cos, sin, pi
import xarray as xr

def spherical(polar_verts):
    phi = polar_verts[:, 0]
    theta = polar_verts[:, 1]
    x = np.cos(phi) * np.cos(theta)
    y = np.sin(phi) * np.cos(theta)
    z = np.sin(theta)
    return np.column_stack([x, y, z])

def load_da(path,variable=None):
    ds = xr.open_dataset(path)
    if variable is not None:
        da = ds[variable]
    else:
        da = ds[list(ds.keys())[0]]
    return da

class data_array():
    def __init__(self, path, name,surface=False,variable=None):
        self.path = path
        self.da = load_da(self.path,variable)
        self.name = name
        self.scene = bpy.context.scene
        if variable != None:
            self.attr_name = variable
        else:
            self.attr_name = name+'_attribute'
        self.surface = surface
        
    def mesh_make(self,verts,attrs,sphere=False):
        if self.name in bpy.data.meshes.keys():
            bpy.data.meshes.remove(bpy.data.meshes[self.name])
        m = bpy.data.meshes.new(self.name)
        if self.surface:
            ysize,xsize = self.da.shape[-2:] ##Potential issues 
            polygons = [(i, i - 1, i - 1 + xsize, i + xsize) for i in range( 1, len(verts) - xsize )  if i % xsize != 0]
            if sphere:
                polygons += [tuple([i for i in range(xsize)])]
                polygons += [(xsize*(i+1),xsize*(i+1)+xsize-1,xsize*i+xsize-1,xsize*i) for i in range(ysize-1)]
                polygons += [tuple([i for i in range(len(verts)-xsize,len(verts))])]

            m.from_pydata(verts,[],polygons)
        else:
            m.from_pydata(verts,[],[])

        o = bpy.data.objects.new(self.name, m )
        self.scene.collection.objects.link(o)

        bpy.data.objects[self.name].data.attributes.new(name=self.attr_name, type='FLOAT', domain='POINT')
        bpy.data.objects[self.name].data.attributes[self.attr_name].data.foreach_set('value', attrs)

    def materials(self):
        self.mat_name = self.name+'_Normalized'
        if self.mat_name in bpy.data.materials.keys():
            bpy.data.materials.remove(bpy.data.materials[self.mat_name])
        bpy.data.materials.new(self.mat_name).use_nodes=True

        nodes = bpy.data.materials[self.mat_name].node_tree.nodes
        links = bpy.data.materials[self.mat_name].node_tree.links

        attr = nodes.new(type='ShaderNodeAttribute')
        cramp = nodes.new(type='ShaderNodeValToRGB')
        principled = nodes['Principled BSDF']
        
        attr.attribute_name = 'Factor'

        links.new(attr.outputs[2],cramp.inputs[0])
        links.new(cramp.outputs[0],principled.inputs[0])

        attr.location = (-600,200)
        cramp.location = (-300,200)
        
        colorramp = nodes["ColorRamp"].color_ramp
        colorramp.elements.new(0.5)
        elements = colorramp.elements
        elements[0].color = (1,0,0,1)
        elements[1].color = (0,1,0,1)
        elements[2].color = (0,0,1,1)
        
    def add_gn(self):
        self.materials()
        name = self.name
        mod_name = name+'_gn'
        if mod_name in bpy.data.node_groups.keys():
            bpy.data.node_groups.remove(bpy.data.node_groups[mod_name])
        node_tree = bpy.data.node_groups.new(mod_name,type='GeometryNodeTree')
        bpy.data.objects[name].modifiers.new(mod_name,type='NODES')
        bpy.data.objects[name].modifiers[mod_name].node_group = node_tree

        nodes = bpy.data.objects[name].modifiers[mod_name].node_group.nodes
        links = bpy.data.objects[name].modifiers[mod_name].node_group.links
        
        input = nodes.new('NodeGroupInput')
        input.outputs[0].type = 'GEOMETRY'
        input.outputs[0].name = 'Input'
        bpy.data.objects[name].modifiers[mod_name].node_group.inputs.new('NodeSocketGeometry','Geometry')
        bpy.data.objects[name].modifiers[mod_name].node_group.inputs.new('NodeSocketFloat','Attribute')
        bpy.data.objects[name].modifiers[mod_name].node_group.inputs.new('NodeSocketFloat','Displacement')
        bpy.data.objects[name].modifiers[mod_name].node_group.outputs.new('NodeSocketGeometry','Geometry')
        bpy.data.objects[name].modifiers[mod_name].node_group.outputs.new('NodeSocketFloat','Factor')
        att_id = bpy.data.objects[name].modifiers[mod_name].node_group.inputs['Attribute'].identifier
        bpy.data.objects[name].modifiers[mod_name][att_id+'_attribute_name'] = self.attr_name
        bpy.data.objects[name].modifiers[mod_name][att_id+'_use_attribute'] = True

        bpy.data.node_groups[mod_name].inputs[2].default_value = 1
        mod = bpy.data.objects[name].modifiers[mod_name]
        self.reset_to_default(mod, "Displacement")
        
        output = nodes.new('NodeGroupOutput')
        output.inputs[0].type = 'GEOMETRY'
        output.inputs[0].name = 'Output'
        return input,output,nodes,links

    def reset_to_default(self,mod,name):
        input = next(i for i in mod.node_group.inputs if i.name == name)
        mod[input.identifier] = input.default_value    

    def plane_gn(self):
        name = self.name
        mod_name = name+'_gn'
        input,output,nodes,links = self.add_gn()
        bpy.data.objects[name].modifiers[mod_name].node_group.inputs.new('NodeSocketFloat','Scale')
        bpy.data.node_groups[mod_name].inputs[3].default_value = 1
        mod = bpy.data.objects[name].modifiers[mod_name]
        self.reset_to_default(mod, "Scale")
        
        math = nodes.new('ShaderNodeMath')
        math.operation = 'MULTIPLY'
        math.inputs[1].default_value = 1

        div_math = nodes.new('ShaderNodeMath')
        div_math.operation = 'DIVIDE'
        div_math.inputs[0].default_value = 20
        
        comb_xyz = nodes.new('ShaderNodeCombineXYZ')
        attr_stat = nodes.new('GeometryNodeAttributeStatistic')
        map_range = nodes.new('ShaderNodeMapRange')
        
        set_pos = nodes.new('GeometryNodeSetPosition')
        position = nodes.new('GeometryNodeInputPosition')
        vect_math = nodes.new('ShaderNodeVectorMath')
        set_mat = nodes.new('GeometryNodeSetMaterial')
        vect_math.operation = 'DIVIDE'
        
        set_mat.inputs[2].default_value = bpy.data.materials[self.mat_name]

        if self.surface:
            dual_mesh = nodes.new('GeometryNodeDualMesh')
            shade_smooth = nodes.new('GeometryNodeSetShadeSmooth')
            links.new(set_pos.outputs[0],dual_mesh.inputs[0])
            links.new(dual_mesh.outputs[0],shade_smooth.inputs[0])
            links.new(shade_smooth.outputs[0],set_mat.inputs[0])

            dual_mesh.location = (-50,-200)
            shade_smooth.location = (100,-200)

        else:
            to_points = nodes.new('GeometryNodeMeshToPoints')
            links.new(set_pos.outputs[0],to_points.inputs[0])
            links.new(to_points.outputs[0],set_mat.inputs[0])
            to_points.location = (0,0)


        links.new(input.outputs[0],attr_stat.inputs[0])
        links.new(input.outputs[1],attr_stat.inputs[2])
        links.new(input.outputs[3],div_math.inputs[1])
        links.new(attr_stat.outputs[3],map_range.inputs[1])
        links.new(attr_stat.outputs[4],map_range.inputs[2])
        links.new(input.outputs[1],map_range.inputs[0])
        links.new(map_range.outputs[0],output.inputs[1])
        links.new(input.outputs[2],math.inputs[1])
        links.new(input.outputs[0],set_pos.inputs[0])
        links.new(div_math.outputs[0],vect_math.inputs[1])
        links.new(set_mat.outputs[0],output.inputs[0])
        links.new(position.outputs[0],vect_math.inputs[0])
        links.new(vect_math.outputs[0],set_pos.inputs[2])
        links.new(input.outputs[1],math.inputs[0])
        links.new(math.outputs[0],comb_xyz.inputs[-1])
        links.new(comb_xyz.outputs[0],set_pos.inputs[-1])
        
        fac_id = bpy.data.objects[name].modifiers[mod_name].node_group.outputs['Factor'].identifier
        bpy.data.objects[name].modifiers[mod_name][fac_id+'_attribute_name'] = 'Factor'
        bpy.data.objects[name].modifiers[mod_name][fac_id+'_use_attribute'] = True
        
        attr_stat.location = (-300,400)
        map_range.location = (0,400)
        
        
        input.location = (-950,0)
        set_pos.location = (-200,0)
        output.location = (400,0)
        set_mat.location = (200,0)
        div_math.location = (-800,-100)
        position.location = (-800,-260)
        vect_math.location = (-600,-200)
        math.location = (-600,-50)
        comb_xyz.location = (-400,-50)   
    
    def sphere_gn(self):
        name = self.name
        input,output,nodes,links = self.add_gn()
        mod_name = name+'_gn'
        bpy.data.objects[name].modifiers[mod_name].node_group.inputs.new('NodeSocketFloat','Radius')
        bpy.data.node_groups[mod_name].inputs[3].default_value = 10
        mod = bpy.data.objects[name].modifiers[mod_name]
        self.reset_to_default(mod, "Radius")
        
        math = nodes.new('ShaderNodeMath')
        math.operation = 'MULTIPLY'
        math.inputs[1].default_value = 1
        
        attr_stat = nodes.new('GeometryNodeAttributeStatistic')
        map_range = nodes.new('ShaderNodeMapRange')
        set_pos = nodes.new('GeometryNodeSetPosition')
        position = nodes.new('GeometryNodeInputPosition')
        vect_disp = nodes.new('ShaderNodeVectorMath')
        vect_pos = nodes.new('ShaderNodeVectorMath')
        vect_disp.operation = 'MULTIPLY'
        vect_pos.operation = 'MULTIPLY'
        set_mat = nodes.new('GeometryNodeSetMaterial')
        
        if self.surface:
            dual_mesh = nodes.new('GeometryNodeDualMesh')
            shade_smooth = nodes.new('GeometryNodeSetShadeSmooth')
            links.new(set_pos.outputs[0],dual_mesh.inputs[0])
            links.new(dual_mesh.outputs[0],shade_smooth.inputs[0])
            links.new(shade_smooth.outputs[0],set_mat.inputs[0])

            dual_mesh.location = (-50,-200)
            shade_smooth.location = (100,-200)

        else:
            to_points = nodes.new('GeometryNodeMeshToPoints')
            links.new(set_pos.outputs[0],to_points.inputs[0])
            links.new(to_points.outputs[0],set_mat.inputs[0])
            to_points.location = (0,0)
        
        set_mat.inputs[2].default_value = bpy.data.materials[self.mat_name]
        
        links.new(input.outputs[0],attr_stat.inputs[0])
        links.new(input.outputs[1],attr_stat.inputs[2])
        links.new(attr_stat.outputs[3],map_range.inputs[1])
        links.new(attr_stat.outputs[4],map_range.inputs[2])
        links.new(input.outputs[1],map_range.inputs[0])
        links.new(map_range.outputs[0],output.inputs[1])
        
        links.new(input.outputs[0],set_pos.inputs[0])
        links.new(set_mat.outputs[0],output.inputs[0])
        links.new(input.outputs[1],math.inputs[0])
        links.new(input.outputs[2],math.inputs[1])
        links.new(math.outputs[0],vect_disp.inputs[0])
        links.new(input.outputs[3],vect_pos.inputs[1])
        links.new(position.outputs[0],vect_pos.inputs[0])
        links.new(position.outputs[0],vect_disp.inputs[1])
        links.new(vect_pos.outputs[0],set_pos.inputs[2])
        links.new(vect_disp.outputs[0],set_pos.inputs[-1])
        
        fac_id = bpy.data.objects[name].modifiers[mod_name].node_group.outputs['Factor'].identifier
        bpy.data.objects[name].modifiers[mod_name][fac_id+'_attribute_name'] = 'Factor'
        bpy.data.objects[name].modifiers[mod_name][fac_id+'_use_attribute'] = True
        
        attr_stat.location = (-300,400)
        map_range.location = (0,400)
        
        input.location = (-800,0)
        set_pos.location = (-200,0)
        output.location = (400,0)
        set_mat.location = (200,0)
        position.location = (-800,-400)
        math.location = (-700,-200)
        vect_pos.location = (-400,-200)
        vect_disp.location = (-400,-400)
    

    def construct_plane(self,y_increase=True):
        if self.name in self.scene.objects.keys():
            return
        if len(self.da.shape) == 3:
            array = self.da[0].values
        elif len(self.da.shape) == 2:
            array = self.da.values
        else:
            print('Cannot be larger than three dimensions. Found:',len(self.da.shape))
            return
        if self.surface and np.isnan(array).any():
            array = np.nan_to_num(array)
        array1 = np.expand_dims(array,0)
        verts = np.moveaxis(np.indices(array1.shape),0,-1)
        verts = np.flip(verts,-1)
        verts = verts.reshape(np.prod(verts.shape[:-1]),3)
        flat = array.flatten()
        verts = verts[~np.isnan(flat)]
        attrs = flat[~np.isnan(flat)]
        if not y_increase:
            verts *= [1,-1,1]
            verts+= [0,array.shape[0]-1,0]
        self.mesh_make(verts,attrs)
        self.plane_gn()


    def construct_sphere(self,use_coords=True,y_increase=True):
        if self.name in self.scene.objects.keys():
            return
        if len(self.da.shape) == 3:
            array = self.da[0]
        elif len(self.da.shape) == 2:
            array = self.da
        else:
            print('Cannot be larger than three dimensions. Found:',len(self.da.shape))
            return
        if self.surface and np.isnan(array).any():
            array = np.nan_to_num(array)
        else:
            array = array.values
        flat = array.flatten()
        attrs = flat[~np.isnan(flat)]
        if use_coords:
            coords = [i.lower() for i in list(self.da.coords)]
            for i in ['lat','lats','latitude','latitudes','y']:
                if i in coords:
                    lats = self.da[i].values
                    break
            for i in ['lon','lons','longitude','longitudes','x']:
                if i in coords:
                    lons = self.da[i].values
                    break
            lats,lons = np.meshgrid(lats,lons)
            lats = np.moveaxis(lats[...,np.newaxis],0,1)
            lons = np.moveaxis(lons[...,np.newaxis],0,1)
            coords = np.concatenate((lons,lats),axis=-1)
            coords = coords.reshape(np.prod(coords.shape[:-1]),2)
            coords = coords[~np.isnan(flat)]
            coords = coords*(pi/180)
            polar_verts = spherical(coords)
        else:
            array1 = np.expand_dims(array,0)
            verts = np.moveaxis(np.indices(array1.shape),0,-1)
            verts = np.flip(verts,-1)
            verts = verts.reshape(np.prod(verts.shape[:-1]),3)
            if not y_increase:
                verts *= [1,-1,1]
                verts += [0,array.shape[0]-1,0]
            flat = array.values.flatten()
            verts = verts[~np.isnan(flat)]    
            polar_verts = (verts*[360/array.shape[1],180/array.shape[0],1]).astype('float16')
            polar_verts -= [180,90,0]
            polar_verts = polar_verts*(pi/180)
            polar_verts = spherical(polar_verts)
            
        self.mesh_make(polar_verts,attrs,sphere=True)
        self.sphere_gn()

    def update(self,scene,depsgraph):
        assert len(self.da.shape) == 3, 'Animation Requires three dimenisions'
        currentFrame = scene.frame_current
        slice = self.da[currentFrame-1].values
        if not self.surface:
            slice = slice[~np.isnan(slice)]
        bpy.data.objects[self.name].data.attributes[self.attr_name].data.foreach_set('value', slice.flatten())
        bpy.data.objects[self.name].update_tag()
        
    def add_attr(variable,animate=None):
        da = load_da(self.path,variable=variable)
        if len(da.shape) == 3:
            array = da[0]
        elif len(da.shape) == 2:
            array = da
        flat = da.values.flatten()
        attrs = flat[~np.isnan(flat)]
        bpy.data.objects[self.name].data.attributes.new(name=variable, type='FLOAT', domain='POINT')
        bpy.data.objects[self.name].data.attributes[variable].data.foreach_set('value', attrs)

if __name__ == '__main__':
    
    name = 'forest_age'
    path = 'H:/Work Blender/carboscope.nc'

    chirps = data_array(path,name,surface=True,variable='co2flux_land')
    chirps.construct_plane()
    ## Uncomment the below line when you are happy with the plot and run the script again for animations
    #bpy.app.handlers.frame_change_pre.append(chirps.update)