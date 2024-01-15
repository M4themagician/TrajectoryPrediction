from dynamics import TrajectoryNetwork, SingleTrackDataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch


def demo():
    batch_size = 1
    num_plots = 5
    print_n = 500 * batch_size
    input_steps = 150
    forecast_steps = 30

    dataset = SingleTrackDataset(input_steps, forecast_steps, print_n)
    net = TrajectoryNetwork(input_dim=8, state_dim=6, input_timesteps=input_steps, forecast_steps=forecast_steps)
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load("weights.pth"))
    net = net.module.cuda()
    fig, ax = plt.subplots(5, 5, layout='compressed')

    for i in range(num_plots*num_plots):
        j = i % num_plots
        k = i // num_plots
        x, y = dataset.__getitem__(0)
        x = x.cuda().unsqueeze(0)
        y = y.cuda().unsqueeze(0)
        prediction = net(torch.flatten(x, start_dim=1, end_dim=2))
        input_trajectory = x[0].detach().squeeze(0).cpu().numpy()
        future_gt = y[0].detach().squeeze(0).cpu().numpy()
        future_predicted = torch.unflatten(prediction[0].detach().squeeze(0), dim=0, sizes=(6, forecast_steps)).cpu().numpy()
        input_x = np.cumsum(input_trajectory[0, :])
        input_y = np.cumsum(input_trajectory[1, :])
        ax[j, k].plot(input_x, input_y, label="input")
        ax[j, k].plot(np.cumsum(future_gt[0, :]) + input_x[-1], np.cumsum(future_gt[1, :]) + input_y[-1], label="gt")
        ax[j, k].plot(np.cumsum(future_predicted[0, :]) + input_x[-1], np.cumsum(future_predicted[1, :]) + input_y[-1], label="prediction")
        ax[j, k].legend()
    plt.show()


def model_predictive_simulation():
    batch_size = 1
    num_plots = 10
    num_steps = 150
    print_n = 500 * batch_size
    input_steps = 150
    forecast_steps = 30

    dataset = SingleTrackDataset(input_steps, forecast_steps, print_n)
    net = TrajectoryNetwork(input_dim=8, state_dim=6, input_timesteps=input_steps, forecast_steps=forecast_steps)
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load("weights_.pth"))
    net = net.module.cuda()
    for _ in range(num_plots):
        x, y = dataset.__getitem__(0)
        x = x.cuda().unsqueeze(0)
        y = y.cuda().unsqueeze(0)
        prediction = net(torch.flatten(x, start_dim=1, end_dim=2))
        input_trajectory = x[0].detach().squeeze(0).cpu().numpy()
        for i in range(num_steps):
            future_gt = y[0].detach().squeeze(0).cpu().numpy()
            future_predicted = torch.unflatten(prediction[0].detach().squeeze(0), dim=0, sizes=(6, forecast_steps)).cpu().numpy()
            input_x = np.cumsum(input_trajectory[0, :])
            input_y = np.cumsum(input_trajectory[1, :])
            input_trajectory[:, :-1] = input_trajectory[:, 1:]
            input_trajectory[:6, -1] = future_predicted[:, 0]
            x = torch.from_numpy(input_trajectory).cuda().unsqueeze(0)
            prediction = net(torch.flatten(x, start_dim=1, end_dim=2))
        plt.plot(input_x, input_y, label="input")
        # plt.plot(np.cumsum(future_gt[0, :]) + input_x[-1], np.cumsum(future_gt[1, :]) + input_y[-1], label="gt")
        plt.plot(np.cumsum(future_predicted[0, :]) + input_x[-1], np.cumsum(future_predicted[1, :]) + input_y[-1], label="prediction")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    demo()
