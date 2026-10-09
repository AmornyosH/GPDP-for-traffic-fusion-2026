import os
import time
import tempfile
import numpy as np
import traci

from agents.AV import AutonomousVehicle
from environments.double_ring import DoubleRing


VIEW_ID = 'View #0'


def recordSimulationGif(dataset: dict,
                        num_veh: int = 44,
                        save_path: str = 'figures/simulation.gif',
                        imposter: bool = False,
                        capture_every: int = 5,
                        gif_fps: int = 10,
                        record_steps: int = 500,
                        resize_width: int = 640):
    """
    Opens sumo-gui, runs warmup, then captures frames into an animated GIF.

    Requirements: Pillow (pip install Pillow), SUMO with sumo-gui binary.

    Args:
        dataset:       Offline dataset dict (needed to initialise GPDP agent).
        num_veh:       Number of vehicles in the ring road.
        save_path:     Output path for the GIF file.
        imposter:      If True, trigger the dynamics-shift imposter vehicle.
        capture_every: Capture one frame every N simulation steps.
        gif_fps:       Playback speed of the output GIF (frames per second).
        record_steps:  Number of simulation steps to record (after warmup).
        resize_width:  Resize each frame to this pixel width (keeps aspect ratio).
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError('Pillow is required for GIF recording: pip install Pillow')

    cond = 'shifted' if imposter else 'standard'
    env = DoubleRing(render=True, cond=cond)
    horizon = env.params_dict['horizon']

    agent = AutonomousVehicle(load=True, params_dict=env.params_dict,
                              dataset=dataset, id='AV_1', online=False,
                              num_veh=num_veh)

    frames_dir = tempfile.mkdtemp()
    frame_paths = []

    print('(record_gif) Running warmup — please wait...')
    env.reset(veh=agent, num_veh=num_veh)
    obs = agent.getObservation()

    try:
        traci.gui.setZoom(VIEW_ID, 500)
    except Exception:
        pass

    _impos_veh_id = env.initImposter(av_lane=obs[1])
    n_steps = min(record_steps, horizon)

    print(f'(record_gif) Capturing {n_steps} steps (every {capture_every}) → '
          f'~{n_steps // capture_every} frames at {gif_fps} fps.')

    for t in range(n_steps):
        _action = agent.actionPerform(state=obs, s_print=False, pred=True)
        obs, _reward, _, _ = env.step(veh=agent)

        if imposter and t >= n_steps * 0.2:
            env.setImposterSpeed(id=_impos_veh_id)

        if t % capture_every == 0:
            frame_path = os.path.join(frames_dir, f'frame_{t:05d}.png')
            try:
                traci.gui.screenshot(VIEW_ID, frame_path)
                frame_paths.append(frame_path)
            except Exception as e:
                print(f'  screenshot failed at step {t}: {e}')

    env.close()
    time.sleep(0.5)  # allow last screenshots to flush

    print(f'(record_gif) Building GIF from {len(frame_paths)} captured frames...')

    images = []
    for p in frame_paths:
        if not os.path.isfile(p):
            continue
        img = Image.open(p)
        if resize_width and img.width > resize_width:
            ratio = resize_width / img.width
            img = img.resize((resize_width, int(img.height * ratio)), Image.LANCZOS)
        images.append(img.convert('RGB'))

    if not images:
        print('(record_gif) No frames captured — is sumo-gui accessible?')
        return

    gif_frames = [img.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
                  for img in images]

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    gif_frames[0].save(
        save_path,
        save_all=True,
        append_images=gif_frames[1:],
        loop=0,
        duration=int(1000 / gif_fps),
    )

    size_kb = os.path.getsize(save_path) // 1024
    print(f'(record_gif) Saved: {save_path}  ({len(gif_frames)} frames, {size_kb} KB)')

    for p in frame_paths:
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(frames_dir)
    except OSError:
        pass
