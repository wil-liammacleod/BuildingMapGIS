import trimesh
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run whiteboxtest/view_glb.py <path_to_glb>")
        sys.exit(1)
        
    glb_path = sys.argv[1]
    if not os.path.exists(glb_path):
        print(f"Error: file '{glb_path}' does not exist.")
        sys.exit(1)
        
    print(f"Loading GLB from: {glb_path}")
    mesh = trimesh.load(glb_path)
    
    print("\n--- Mesh Information ---")
    if isinstance(mesh, trimesh.Scene):
        print(f"Loaded a scene with {len(mesh.geometry)} geometries.")
        for name, geom in mesh.geometry.items():
            print(f"  Geom: {name} | Vertices: {len(geom.vertices)} | Faces: {len(geom.faces)}")
    else:
        print(f"Vertices: {len(mesh.vertices)}")
        print(f"Faces: {len(mesh.faces)}")
        print(f"Bounding Box: {mesh.bounds}")
        print(f"Is watertight: {mesh.is_watertight}")
        
    print("\nOpening 3D viewer window. Close it to finish.")
    mesh.show()

if __name__ == "__main__":
    main()
