import h5py

file_path = r"D:\HUST\Python-Project\demo_v15.hdf5"

with h5py.File(file_path, "r") as f:
    print("=== THÔNG TIN TỔNG QUAN ===")
    print("Các keys chính:", list(f.keys()))
    
    data_group = f["data"]
    episodes = list(data_group.keys())
    print(f"Tổng số episode: {len(episodes)}")
    
    # Xem chi tiết kích thước dữ liệu của một vài episode đầu tiên
    for ep_name in episodes[:3]: # Duyệt qua 3 episode đầu tiên làm ví dụ
        ep = data_group[ep_name]
        print(f"\n--- Episode: {ep_name} ---")
        print("  - States shape:", ep["states"].shape)
        print("  - Actions shape:", ep["actions"].shape)
        
        # Nếu bạn muốn xem thêm các thông tin controller hoặc thông số khác
        if "controller_info" in ep:
            print("  - Có dữ liệu controller_info")