
last layer: conv_preds

x = Conv2D(classes, (1, 1),
                   padding='same', name='conv_preds')(x)