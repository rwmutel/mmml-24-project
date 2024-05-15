import sys

sys.path.append("core")

import argparse
import csv
import os
import time

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from utils import flow_viz, frame_utils
from utils.utils import InputPadder, forward_interpolate

from raft import RAFT


@torch.no_grad()
def create_sintel_submission(
    model, iters=32, warm_start=False, output_path="sintel_submission"
):
    """Create submission for the Sintel leaderboard"""
    model.eval()
    for dstype in tqdm(["clean", "final"]):
        test_dataset = datasets.MpiSintel(split="test", aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        for test_id in tqdm(range(len(test_dataset))):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(
                image1, image2, iters=iters, flow_init=flow_prev, test_mode=True
            )
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, "frame%04d.flo" % (frame + 1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def validate_sintel(model, iters=32):
    """Perform validation using the Sintel (train) split"""
    model.eval()
    results = {}

    # checkpoint_name = kwargs.get("checkpoint_name", "")
    # if checkpoint_name:
    #     file_name = f"{checkpoint_name.split('/')[1].split('.')[0]}"
    # output_txt = (
    #         f"sintel_results_{file_name}.txt"
    #         if checkpoint_name
    #         else "sintel_results.txt"
    #     )
    # output_csv = (
    #     f"sintel_results_{file_name}.csv"
    #     if checkpoint_name
    #     else "sintel_results.csv"
    # )

    output_txt = "sintel_results_RAFT.txt"
    output_csv = "sintel_results_RAFT.csv"
    memory_consumption = []
    inference_times = []

    for dstype in tqdm(["clean", "final"]):
        val_dataset = datasets.MpiSintel(split="training", dstype=dstype)
        epe_list = []

        # for val_id in tqdm(range(len(val_dataset))):
        for val_id in tqdm(range(10)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)

            end_event.record()
            torch.cuda.synchronize()
            inference_time = (
                start_event.elapsed_time(end_event) / 1000.0
            )  # Convert to seconds

            memory_consumption.append(
                torch.cuda.max_memory_allocated() / 1024**2
            )  # Memory consumption in MB
            inference_times.append(inference_time)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        print(
            "Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f"
            % (dstype, epe, px1, px3, px5)
        )
        print(
            f"Validation ({dstype}) EPE: {epe:.3f}, 1px: {px1:.2f}, 3px: {px3:.2f}, 5px: {px5:.2f}"
        )

        with open(output_txt, "a") as f:
            f.write(
                f"Validation ({dstype}) EPE: {epe:.3f}, 1px: {px1:.2f}, 3px: {px3:.2f}, 5px: {px5:.2f}\n"
            )
        results[dstype] = np.mean(epe_list)

        output_csv = f"sintel_results_RAFT_{dstype}.csv"
        # if checkpoint_name
        # else "sintel_results.csv"
        # )

        save_csv(
            output_file=output_csv,
            memory_consumption=memory_consumption,
            inference_times=inference_times,
            epe_values=epe_all,
            # rho_values=rho_list,
        )

    return results


def save_csv(output_file, memory_consumption, inference_times, epe_values):
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Memory Consumption (MB)", "Inference Time (s)", "EPE"])

        for mem, inf, epe in zip(memory_consumption, inference_times, epe_values):
            writer.writerow([mem, inf, epe])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="restore checkpoint")
    parser.add_argument("--dataset", help="dataset for evaluation")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    parser.add_argument(
        "--alternate_corr",
        action="store_true",
        help="use efficent correlation implementation",
    )
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if args.dataset == "chairs":
            validate_chairs(model.module)

        elif args.dataset == "sintel":
            validate_sintel(model.module)

        elif args.dataset == "kitti":
            validate_kitti(model.module)
