style_layers = ['r11','r21','r31','r41', 'r51'] 
loss_fn_style = GramMSELoss().to(device)
style_targets = [gram_matrix(feature) for feature in vgg19(style_img, style_layers)]