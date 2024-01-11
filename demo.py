from dynamics import TrajectoryNetwork, SingleTrackDataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch


def demo():
    batch_size = 1
    num_plots = 10
    print_n = 500 * batch_size
    input_steps = 150
    forecast_steps = 30

    dataset = SingleTrackDataset(input_steps, forecast_steps, print_n)
    net = TrajectoryNetwork(input_dim=8, state_dim=6, input_timesteps=input_steps, forecast_steps=forecast_steps)
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load("weights.pth"))
    net = net.module.cuda()
    for _ in range(num_plots):
        x, y = dataset.__getitem__(0)
        x = x.cuda().unsqueeze(0)
        y = y.cuda().unsqueeze(0)
        prediction = net(torch.flatten(x, start_dim=1, end_dim=2))
        input_trajectory = x[0].detach().squeeze(0).cpu().numpy()
        future_gt = y[0].detach().squeeze(0).cpu().numpy()
        future_predicted = torch.unflatten(prediction[0].detach().squeeze(0), dim=0, sizes=(6, forecast_steps)).cpu().numpy()
        input_x = np.cumsum(input_trajectory[0, :])
        input_y = np.cumsum(input_trajectory[1, :])
        plt.plot(input_x, input_y, label="input")
        plt.plot(np.cumsum(future_gt[0, :]) + input_x[-1], np.cumsum(future_gt[1, :]) + input_y[-1], label="gt")
        plt.plot(np.cumsum(future_predicted[0::6]) + input_x[-1], np.cumsum(future_predicted[1::6]) + input_y[-1], label="prediction")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    demo()