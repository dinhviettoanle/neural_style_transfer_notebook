style_layers = ['r12','r22','r33','r43'] 
style_targets = [gram_matrix(feature) for feature in vgg16(style, style_layers)]
loss_fn_style = GramMSELoss().to(device)

content_layers = ['r22']
loss_fn_content = nn.MSELoss().to(device)

loss_layers = style_layers + content_layers