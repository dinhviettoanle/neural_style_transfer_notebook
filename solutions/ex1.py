content_layers = ['r42']
loss_fn_content = nn.MSELoss().to(device)
content_targets = [feature for feature in vgg19(content_img, content_layers)]