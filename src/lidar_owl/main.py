import open3d as o3d
import numpy as np

def main():
    print("Welcome to the Lidar Owl application!")
    # Test Open3D import by creating and printing a PointCloud object
    # Erstelle eine zufällige Punktwolke mit 100 Punkten
    points = np.random.rand(100, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    print("Open3D PointCloud object created:", np.asarray(pcd.points))

if __name__ == "__main__":
    main()