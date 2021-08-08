import torch
import torch.nn as nn
from fasgan.gencore import Core

class Generator(nn.Module):

	def __init__(self, latent_dim=100, image_dim=64, num_channels=3):
		super(Generator, self).__init__()

		self.latent_dim = latent_dim
		self.image_dim = image_dim
		self.num_channels = num_channels

		self.conv0 = nn.ConvTranspose2d(self.latent_dim, self.image_dim*16, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(self.image_dim*16)
		self.relu = nn.ReLU(inplace=True)

		self.block1 = nn.Sequential(Core(self.image_dim*16, self.image_dim*8, 4, 2, 1))
		self.inflate1 = nn.Sequential(
				nn.ConvTranspose2d(self.image_dim*4, self.image_dim*8, kernel_size=2, stride=2, bias=False),
				nn.BatchNorm2d(self.image_dim*8),
				nn.ReLU(inplace=True),
			)

		self.block2 = nn.Sequential(Core(self.image_dim*8, self.image_dim*4, 4, 2, 1))
		self.inflate2 = nn.Sequential(
				nn.ConvTranspose2d(self.image_dim*2, self.image_dim*4, kernel_size=2, stride=2, bias=False),
				nn.BatchNorm2d(self.image_dim*4),
				nn.ReLU(inplace=True),
			)

		self.block3 = nn.Sequential(Core(self.image_dim*4, self.image_dim*2, 1, 1, 0))
		self.conv1 = nn.ConvTranspose2d(self.image_dim, self.num_channels, kernel_size=1, stride=1, bias=False)

		self.tanh = nn.Tanh()

	def forward(self, input):

		out = self.conv0(input)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.block1(out)
		out = self.inflate1(out)

		out = self.block2(out)
		out = self.inflate2(out)

		out = self.block3(out)
		out = self.conv1(out)
		out = self.tanh(out)

		return out