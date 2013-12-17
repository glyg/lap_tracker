import bpy

def hex_to_rgb(col, factors = 255.):
    return tuple(c / factors for c in bytes.fromhex(col.replace('#', '')))


def set_moving_sphere(name, xx, yy, zz, color,
                      timelapse, frame_rate, speed, radius=0.25):
    
    size = (radius, radius, radius)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, size=1, view_align=False,
    enter_editmode=False, location=(0, 0, 0), rotation=(0, 0, 0))
    obj = bpy.context.object

    obj.name = name
    obj.dimensions = size

    material = bpy.data.materials.new("%s_color" % name)
    material.diffuse_color = hex_to_rgb(color)
    bpy.ops.object.material_slot_add()
    slot = obj.material_slots[0]
    slot.material = material

    for t, x, y, z in zip(timelapse, xx, yy, zz):
        obj.location = (x, y, z)
        obj.keyframe_insert('location', frame=t * (frame_rate / speed))

    T_START = timelapse[0] * (frame_rate / speed)
    T_STOP = timelapse[-1] * (frame_rate / speed)


    
    ###### place at t = T_START-1 a keyframe with hide set to True
    current_frame = T_START - 1
    bpy.context.scene.frame_set(frame=current_frame)
    bpy.context.active_object.hide = True
    bpy.context.active_object.keyframe_insert(data_path="hide", 
                                              index=-1, 
                                              frame=current_frame)    

    ###### place at T_START a keyframe with hide set to False
    current_frame = T_START
    bpy.context.scene.frame_set(frame=current_frame)
    bpy.context.active_object.hide = False
    bpy.context.active_object.keyframe_insert(  data_path="hide", 
                                                index=-1, 
                                                frame=current_frame)

    ###### place at T_STOP a keyframe with hide set to False
    current_frame = T_STOP
    bpy.context.scene.frame_set(frame=current_frame)
    bpy.context.active_object.hide = False
    bpy.context.active_object.keyframe_insert(  data_path="hide", 
                                                index=-1, 
                                                frame=current_frame)

    ###### place at t = T_STOP + 1 a keyframe with hide set to True
    current_frame = T_STOP + 1
    bpy.context.scene.frame_set(frame=current_frame)
    bpy.context.active_object.hide = True
    bpy.context.active_object.keyframe_insert(data_path="hide", 
                                              index=-1, 
                                              frame=current_frame)    


        
    return obj
