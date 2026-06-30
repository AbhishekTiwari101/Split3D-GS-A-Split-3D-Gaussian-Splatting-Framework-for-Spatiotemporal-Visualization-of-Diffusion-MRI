model = SplitGaussianSplatter(num_splats=30000, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for epoch in range(100):
    rendered = model(dwi_volume, fa_map)
    loss = model.compute_total_loss(dwi_volume, rendered)
    loss.backward()
    optimizer.step()
