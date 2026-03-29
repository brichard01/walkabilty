import os, cv2
base = r"C:/Abderrahim Internship Data/Data/P09/semantic_segmentation/masks"
for fidx in [202,203,204,205]:
    p1 = os.path.join(base, f"mask_{fidx:05d}.png")
    p2 = os.path.join(base, f"mask_{fidx}.png")
    path = p1 if os.path.exists(p1) else p2
    print(fidx, "exists?", os.path.exists(path), path)
    if os.path.exists(path):
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print("   mask shape:", m.shape)  # (h,w)