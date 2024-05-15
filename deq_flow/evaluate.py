import sys

sys.path.append("core")

import csv
import os

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from utils import frame_utils
from utils.utils import InputPadder, forward_interpolate

MAX_FLOW = 400


@torch.no_grad()
def create_sintel_submission(
    model,
    warm_start=False,
    fixed_point_reuse=False,
    output_path="sintel_submission",
    **kwargs,
):
    """Create submission for the Sintel leaderboard"""
    model.eval()
    for dstype in ["clean", "final"]:
        test_dataset = datasets.MpiSintel(split="test", aug_params=None, dstype=dstype)

        sequence_prev, flow_prev, fixed_point = None, None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
                fixed_point = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr, info = model(
                image1, image2, flow_init=flow_prev, cached_result=fixed_point, **kwargs
            )
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            # You may choose to use some hacks here,
            # for example, warm start, i.e., reusing the f* part with a borderline check (forward_interpolate),
            # which was orignally taken by RAFT.
            # This trick usually (only) improves the optical flow estimation on the ``ambush_1'' sequence,
            # in terms of clearer background estimation.
            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            # Note that the fixed point reuse usually does not improve performance.
            # It facilitates the convergence.
            # To improve performance, the borderline check like ``forward_interpolate'' is necessary.
            if fixed_point_reuse:
                net, flow_pred_low = info["cached_result"]
                flow_pred_low = forward_interpolate(flow_pred_low[0])[None].cuda()
                fixed_point = (net, flow_pred_low)

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, "frame%04d.flo" % (frame + 1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def validate_sintel(model, **kwargs):
    """Peform validation using the Sintel (train) split"""
    model.eval()
    best = kwargs.get("best", {"clean-epe": 1e8, "final-epe": 1e8})
    results = {}

    checkpoint_name = kwargs.get("checkpoint_name", "")

    output_txt = (
        f"sintel_results_{checkpoint_name}.txt"
        if checkpoint_name
        else "sintel_results.txt"
    )
    output_csv = (
        f"sintel_results_{checkpoint_name}.csv"
        if checkpoint_name
        else "sintel_results.csv"
    )

    memory_consumption = []
    inference_times = []

    for dstype in tqdm(["clean", "final"]):
        val_dataset = datasets.MpiSintel(split="training", dstype=dstype)
        epe_list = []
        rho_list = []
        info = {"sradius": None, "cached_result": None}

        for val_id in tqdm(range(len(val_dataset))):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            flow_low, flow_pr, info = model(image1, image2, **kwargs)

            end_event.record()
            torch.cuda.synchronize()
            inference_time = (
                start_event.elapsed_time(end_event) / 1000.0
            )  # Convert to seconds

            memory_consumption.append(
                torch.cuda.memory_allocated() / 1024**2
            )  # Memory consumption in MB
            inference_times.append(inference_time)

            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())
            rho_list.append(info["sradius"].mean().item())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1) * 100
        px3 = np.mean(epe_all < 3) * 100
        px5 = np.mean(epe_all < 5) * 100

        best[dstype + "-epe"] = min(epe, best[dstype + "-epe"])
        print(
            f"Validation ({dstype}) EPE: {epe:.3f} ({best[dstype+'-epe']:.3f}), 1px: {px1:.2f}, 3px: {px3:.2f}, 5px: {px5:.2f}"
        )

        with open(output_txt, "a") as f:
            f.write(
                f"Validation ({dstype}) EPE: {epe:.3f} ({best[dstype+'-epe']:.3f}), 1px: {px1:.2f}, 3px: {px3:.2f}, 5px: {px5:.2f}\n"
            )
        results[dstype] = np.mean(epe_list)

        if np.mean(rho_list) != 0:
            print("Spectral radius (%s): %.2f" % (dstype, np.mean(rho_list)))
            with open(output_txt, "a") as f:
                f.write("Spectral radius (%s): %.2f\n" % (dstype, np.mean(rho_list)))

    save_csv(
        output_file=output_csv,
        memory_consumption=memory_consumption,
        inference_times=inference_times,
        epe_values=epe_all,
        rho_values=rho_list,
    )

    return results


def save_csv(output_file, memory_consumption, inference_times, epe_values, rho_values):
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Memory Consumption (MB)", "Inference Time (s)", "EPE", "Rho"])

        for mem, inf, epe, rho in zip(
            memory_consumption, inference_times, epe_values, rho_values
        ):
            writer.writerow([mem, inf, epe, rho])
