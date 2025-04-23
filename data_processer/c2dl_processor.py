import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import center_of_mass
from collections import defaultdict

def parse_man_track_from_masks(track_folder):
    """
    Parses trajectories from man_trackXXX.tif files.
    Returns:
        - trajectories: list of [(x, y, t), ...] per object
    """
    trajectories_dict = defaultdict(list)

    tif_files = sorted([f for f in os.listdir(track_folder) if f.startswith('man_track') and f.endswith('.tif')])
    for t, fname in enumerate(tqdm(tif_files, desc="Parsing masks")):
        frame = np.array(Image.open(os.path.join(track_folder, fname)))
        ids = np.unique(frame)
        ids = ids[ids != 0]  # Ignore background

        for obj_id in ids:
            mask = (frame == obj_id)
            cy, cx = center_of_mass(mask)
            trajectories_dict[obj_id].append((float(cx), float(cy), t))

    return list(trajectories_dict.values())

def build_destinations(trajectories):
    destinations = []
    for traj in trajectories:
        x, y, t = traj[-1]
        destinations.append([(x, y, t)])
    return destinations

def save_c2dl_npy(trajectories, destinations, obstacles, out_path, time_unit=0.10):
    meta_data = {
        'version': 'v2.2',
        'time_unit': time_unit,
        'source': 'c2dl_dataset'
    }
    data = np.array((meta_data, trajectories, destinations, obstacles), dtype=object)
    np.save(out_path, data)
    print(f"Saved: {out_path}")

def process_c2dl_folder(folder_path, output_path, chunk_size=100, time_unit=0.10):
    track_folder = os.path.join(folder_path, '01_GT', 'TRA')
    trajectories = parse_man_track_from_masks(track_folder)
    destinations = build_destinations(trajectories)

    max_time = max([pt[2] for traj in trajectories for pt in traj])
    obstacles = [[1e4, 1e4]]  # dummy obstacle for compatibility

    os.makedirs(output_path, exist_ok=True)
    file_list = []

    for t_start in range(0, max_time + 1, chunk_size):
        t_end = min(t_start + chunk_size, max_time + 1)
        chunk_trajs, chunk_dests = [], []

        for traj, dests in zip(trajectories, destinations):
            chunk = [pt for pt in traj if t_start <= pt[2] < t_end]
            if chunk:
                chunk_trajs.append(chunk)
                chunk_dests.append(dests)

        if not chunk_trajs:
            continue

        out_file = f"c2dl_Dataset_time{t_start:03}-{t_end:03}_timeunit{time_unit:.2f}.npy"
        save_c2dl_npy(chunk_trajs, chunk_dests, obstacles, os.path.join(output_path, out_file), time_unit)
        file_list.append(out_file)

    return file_list

# -------- Main usage --------
if __name__ == "__main__":
    c2dl_root = '/home/ugp25/Divyansh_SPdif/BF-C2DL-HSC'  # Path to folder with 01_GT/TRA/
    output_npy_dir = '/home/ugp25/Divyansh_SPdif/SPDiff/data_origin/C2DL_dataset'
    process_c2dl_folder(c2dl_root, output_npy_dir, chunk_size=100, time_unit=0.10)
