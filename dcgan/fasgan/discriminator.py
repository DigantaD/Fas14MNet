import torch
import torch.nn as nn
from fasgan.discore import Core

class Discriminator(nn.Module):

	def __init__(self, num_classes=3, image_dim=64):
		super(Discriminator, self).__init__()

		self.num_classes = num_classes
		self.image_dim = image_dim

		self.conv0 = nn.Conv2d(self.num_classes, self.image_dim, kernel_size=3, stride=1, padding=1, bias=False)
		self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

		self.block1 = nn.Sequential(Core(self.image_dim, self.image_dim*2, 3, 2, 1))
		self.dilate1 = nn.Sequential(
				nn.Conv2d(self.image_dim*4, self.image_dim*2, kernel_size=1, stride=1, bias=False),
				nn.BatchNorm2d(self.image_dim*2),
				nn.LeakyReLU(0.2, inplace=True),
			)

		self.block2 = nn.Sequential(Core(self.image_dim*2, self.image_dim*4, 3, 2, 1))
		self.dilate2 = nn.Sequential(
				nn.Conv2d(self.image_dim*8, self.image_dim*4, kernel_size=1, stride=1, bias=False),
				nn.BatchNorm2d(self.image_dim*4),
				nn.LeakyReLU(0.2, inplace=True),
			)


		self.block3 = nn.Sequential(Core(self.image_dim*4, self.image_dim*8, 2, 2, 1))
		self.conv1 = nn.Conv2d(self.image_dim*16, 1, kernel_size=2, stride=1, bias=False)

		self.sigmoid = nn.Sigmoid()


	def forward(self, input):

		out = self.conv0(input)
		out = self.leaky_relu(out)

		out = self.block1(out)
		out = self.dilate1(out)

		out = self.block2(out)
		out = self.dilate2(out)

		out = self.block3(out)
		out = self.conv1(out)
		out = self.sigmoid(out)

		return out