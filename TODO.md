## CKPT

1. 0218: flappy bird attention memory
2. 0215-1248: flappy bird, oasis dit, window_size 30
3. 0224: flappy bird attention memory
4. 0228: minecraft easy
5. 0303: easy predict_v
6. 0310: small dataset 16 epochs

## TODO

## GAN

1. long term 的 evaluation 使用fid，fvd，psnr等都不是很可靠，因此希望训练一个discriminator。loss: gt_image + noised_image, sampled_image。bt_loss or classifier_loss ?

2. 使用gan进行post training or pre training。loss: sft_loss(ptx_loss) + gan_loss + kl_penalty? gan_loss需要进行一段sample，帮助缓解误差累积？