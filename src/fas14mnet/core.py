import torch
import torch.nn as nn


class Core(nn.Module):

	expansion = 4

	def __init__(self, in_channels, out_channels, stride=1):
		super(Core, self).__init__()

		self.layer0 = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
			nn.BatchNorm2d(out_channels), nn.ReLU(),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, bias=False),
			nn.BatchNorm2d(out_channels), nn.ReLU(),
		)

		self.layer1 = nn.Sequential(
			nn.Conv2d(out_channels, out_channels*2, kernel_size=1, stride=1, bias=False),
			nn.BatchNorm2d(out_channels*2), nn.ReLU(),
			nn.Conv2d(out_channels*2, out_channels*4, kernel_size=1, bias=False),
			nn.BatchNorm2d(out_channels*4), nn.ReLU(inplace=True),
		)

		self.relu = nn.ReLU()

	def forward(self, input):

		out = self.layer0(input)
		out = self.layer1(out)
		out = self.relu(out)

		return out