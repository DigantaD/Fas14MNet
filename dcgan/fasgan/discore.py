import torch
import torch.nn as nn

class Core(nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
		super(Core, self).__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding

		self.layer0 = nn.Sequential(
				nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
				nn.BatchNorm2d(self.out_channels),
				nn.LeakyReLU(0.2, inplace=True),
			)

		self.layer1 = nn.Sequential(
				nn.Conv2d(self.out_channels, self.out_channels*2, self.kernel_size, self.stride, self.padding, bias=False),
				nn.BatchNorm2d(self.out_channels*2),
				nn.LeakyReLU(0.2, inplace=True),
			)

	def forward(self, input):

		out = self.layer0(input)
		out = self.layer1(out)

		return out