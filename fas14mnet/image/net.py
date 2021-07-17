import torch
import torch.nn as nn
from fas14mnet.image.core import Core

class Fas14MNet(nn.Module):

	def __init__(self, num_classes, layers=[1, 1, 1]):
		super(Fas14MNet, self).__init__()

		self.in_channels = 32
		self.num_classes = num_classes

		self.conv0 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False)
		self.bn0 = nn.BatchNorm2d(32)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
		self.block1 = self.createBlock(64, layers[0], prune=False)
		self.maxpool1 = nn.MaxPool2d(kernel_size=2)
		self.block2 = self.createBlock(256, layers[1], stride=2, prune=True, hidden=1)
		self.maxpool2 = nn.MaxPool2d(kernel_size=2)
		self.block3 = self.createBlock(512, layers[2], stride=2, prune=True, hidden=2)
		self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
		self.fc = nn.Linear(in_features=2048, out_features=num_classes)


	def createBlock(self, planes, cells, stride=1, prune=False, hidden=0):

		blocks = []

		if prune is True:
			out_plane = int((self.in_channels/Core.expansion)*2)

			if hidden == 1:
				self.prune_layer1 = nn.Sequential(
						nn.Conv2d(planes, out_plane, kernel_size=1, stride=stride, bias=False),
						nn.BatchNorm2d(out_plane),
					)
			elif hidden == 2:
				self.prune_layer2 = nn.Sequential(
						nn.Conv2d(out_plane*2, out_plane, kernel_size=1, stride=stride, bias=False),
						nn.BatchNorm2d(out_plane)
					)

			self.in_channels = out_plane

		blocks.append(Core(self.in_channels, planes, stride))
		self.in_channels = planes * Core.expansion

		for i in range(cells):
			blocks.append(Core(self.in_channels, planes))

		return nn.Sequential(*blocks)

	def forward(self, input):

		out = self.conv0(input)
		out = self.bn0(out)
		out = self.relu(out)
		out = self.maxpool(out)
		out = self.block1(out)
		out = self.maxpool1(out)
		out = self.prune_layer1(out)
		out = self.block2(out)
		out = self.maxpool2(out)
		out = self.prune_layer2(out)
		out = self.block3(out)
		out = self.avgpool(out)
		out = out.view(-1, 2048)
		out = self.fc(out)

		return out