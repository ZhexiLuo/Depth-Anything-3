import torch
from depth_anything_3.api import DepthAnything3

print('Loading model...')
model = DepthAnything3.from_pretrained('depth-anything/DA3NESTED-GIANT-LARGE-1.1')
model = model.to('cuda')

print('Running inference...')
prediction = model.inference(['assets/examples/SOH/000.png'])

print(f'Depth shape: {prediction.depth.shape}')
print(f'Depth range: {prediction.depth.min():.3f} - {prediction.depth.max():.3f} meters')
print(f'Extrinsics shape: {prediction.extrinsics.shape}')
print(f'Intrinsics shape: {prediction.intrinsics.shape}')
print('Test OK!')
