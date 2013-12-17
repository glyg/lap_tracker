import bpy

def unselect_all():
    if bpy.context.selected_objects:
        # Deselect all
        bpy.ops.object.select_all()

def select_all():
    unselect_all()
    bpy.ops.object.select_all()

def select(objs):
    """
    Select a list of objects
    """

    unselect_all()

    for obj in objs:
        obj.select = True

def is_ortho():
    """
    TODO: find a way to modify 3d view
    """
    for s in bpy.context.window.screen.areas:
            if s.type=="VIEW_3D":
                break
    viewPersp = s.spaces[0].region_3d.view_perspective
    if viewPersp == 'PERSP':
        return False
    else:
        return True

def clear_scene():

    bpy.context.scene.layers[0] = True

    # Unhide all objects
    for obj in bpy.data.objects:
        obj.hide = False

    select_all()

    # remove all selected.
    bpy.ops.object.delete()

    # remove materials
    for m in bpy.data.materials:
        m.user_clear()
        bpy.data.materials.remove(m)

    # remove the meshes, they have no users anymore.a
    for item in bpy.data.meshes:
      bpy.data.meshes.remove(item)

    # Reset cursor position
    bpy.context.scene.cursor_location = (0, 0, 0)

    return bpy.context.scene
