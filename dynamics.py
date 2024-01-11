import torch
import torch.nn as nn
import numpy as np
import torch.optim as opt
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt


class TrajectoryNetwork(nn.Module):
    channels = [64, 256, 128]

    def __init__(self, input_dim=8, state_dim=6, input_timesteps=100, forecast_steps=1):
        super().__init__()
        layers = []
        c = input_dim * input_timesteps
        for c_out in self.channels:
            layers.append(nn.Sequential(nn.Linear(c, c_out), nn.ReLU()))
            c = c_out
        layers.append(nn.Linear(c, state_dim * forecast_steps))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SingleTrackDataset:
    wheelbase = 2.0
    position_range = (-40, 40)
    speed_range = (0, 20)
    acc_range = (0, 1)
    delta_t = 1 / 30
    constraints_up = [1e20, 1e20, 1e20, 1e20, 5, 35 * np.pi / 180]
    constraints_low = [-v for v in constraints_up]

    def __init__(self, input_steps, forecast_steps, len):
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.len = len

    def singletrack_rhs(self, state, controls):
        # state is x, y, heading, speed, acceleration, wheel angle
        # controls are change of acc, change of wheel angle
        rhs = np.zeros(6)
        rhs[0] = np.cos(state[2]) * state[3]
        rhs[1] = np.sin(state[2]) * state[3]
        rhs[2] = state[3] / self.wheelbase * np.tan(state[5])
        rhs[3] = state[4]
        rhs[4] = controls[0]
        rhs[5] = controls[1]
        return rhs

    def get_controls(self, t):
        return np.array([0, 0])

    def get_initial_state(self):
        state = np.zeros(6)
        # state[2] = 22/7/4
        # state[3] = 1
        state[0] = np.random.uniform(low=self.position_range[0], high=self.position_range[1])
        state[1] = np.random.uniform(low=self.position_range[0], high=self.position_range[1])
        state[2] = np.random.uniform(low=0, high=2 * np.pi)
        state[3] = np.random.uniform(low=self.speed_range[0], high=self.speed_range[1])
        state[4] = np.random.uniform(low=self.acc_range[0], high=self.acc_range[1])
        state[5] = np.random.uniform(low=self.constraints_low[-1], high=self.constraints_up[-1])
        return state

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # define starting orientation
        trajectory = np.zeros((8, self.input_steps + self.forecast_steps), dtype=np.float32)
        trajectory_cumulative = np.zeros((6, self.input_steps + self.forecast_steps), dtype=np.float32)
        trajectory[:6, 0] = self.get_initial_state()
        trajectory_cumulative[:, 0] = trajectory[:6, 0]
        for i in range(self.input_steps + self.forecast_steps - 1):
            controls = self.get_controls(i * self.delta_t)
            trajectory_cumulative[:, i + 1] = trajectory_cumulative[:, i] + self.delta_t * self.singletrack_rhs(trajectory_cumulative[:, i], controls)
            trajectory[6:, i] = controls

            if i < self.input_steps:
                trajectory[:6, i + 1] = self.delta_t * self.singletrack_rhs(trajectory_cumulative[:, i], controls)
                trajectory[:2, i + 1] += np.random.normal(0, 0.05, (2))
                trajectory[3, i + 1] += np.random.normal(0, np.pi / 180)
                trajectory[4, i + 1] += np.random.normal(0, 0.05)

            else:
                trajectory[:6, i + 1] = self.delta_t * self.singletrack_rhs(trajectory_cumulative[:, i], controls)

        input_trajectory = trajectory[:, : self.input_steps]
        regression_target = trajectory[:6, self.input_steps :]
        return torch.from_numpy(input_trajectory), torch.from_numpy(regression_target)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size):
    print(f"Running dynamics DDP training on rank {rank}.")
    setup(rank, world_size)
    torch.set_num_threads(1)
    device_id = rank % torch.cuda.device_count()

    batch_size = 1
    print_n = 5000 * batch_size
    input_steps = 150
    forecast_steps = 30
    show_predictions = False

    dataset = SingleTrackDataset(input_steps, forecast_steps, print_n)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    net = TrajectoryNetwork(input_dim=8, state_dim=6, input_timesteps=input_steps, forecast_steps=forecast_steps).to(device_id)

    net = DDP(net, device_ids=[device_id])
    loss_fn = nn.SmoothL1Loss()
    optimizer = opt.AdamW(net.parameters(), lr=1e-4 / world_size)
    scheduler = opt.lr_scheduler.MultiStepLR(optimizer, milestones=(1000000, 2000000, 3000000), gamma=1 / 10)
    running_loss = 0
    iteration = 0
    dist.barrier()

    while True:
        for x, y in dataloader:
            optimizer.zero_grad(set_to_none=True)
            x = x.to(device_id)
            y = y.to(device_id)
            prediction = net(torch.flatten(x, start_dim=1, end_dim=2))
            loss = loss_fn(prediction, torch.flatten(y, start_dim=1, end_dim=2))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            iteration += 1
            scheduler.step()
        if rank == 0:
            print(f"Loss after {iteration} iterations: {running_loss/world_size/print_n}")
            running_loss = 0
            torch.save(net.state_dict(), "weights.pth")
            if show_predictions:
                input_trajectory = x[0].detach().squeeze(0).cpu().numpy()
                future_gt = y[0].detach().squeeze(0).cpu().numpy()
                future_predicted = torch.unflatten(prediction[0].detach().squeeze(0), dim=0, sizes=(6, forecast_steps)).cpu().numpy()
                input_x = np.cumsum(input_trajectory[0, :])
                input_y = np.cumsum(input_trajectory[1, :])
                plt.clf()
                plt.plot(input_x, input_y, label="input")
                plt.plot(np.cumsum(future_gt[0, :]) + input_x[-1], np.cumsum(future_gt[1, :]) + input_y[-1], label="gt")
                plt.plot(np.cumsum(future_predicted[0, :]) + input_x[-1], np.cumsum(future_predicted[1, :]) + input_y[-1], label="prediction")
                plt.legend()
                # plt.axis('scaled')
                plt.pause(0.1)


def run(fn, world_size):
    mp.spawn(fn, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    run(train, 20)
